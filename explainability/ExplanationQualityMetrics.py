import random
from typing import List, Dict, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import auc


class ExplanationQualityMetrics:
    def __init__(self, explainer, device='cpu'):
        """
        Initialize explanation quality metrics calculator

        Args:
            explainer: SpamCNNExplainer instance
            device: Device to run computations on
        """
        self.explainer = explainer
        self.model = explainer.model
        self.device = device

    def configure_shap_for_stability(self, l1_reg='num_features(10)', n_samples=100,
                                     feature_perturbation='independent'):
        """
        Configure SHAP explainer settings to address numerical stability warnings

        Args:
            l1_reg: L1 regularization setting for SHAP. Using 'num_features(N)'
                   limits to N features which improves stability
            n_samples: Number of samples for SHAP. Higher values improve stability
            feature_perturbation: Method for handling feature dependencies

        Returns:
            True if configuration was successful
        """
        try:
            # Configure the underlying SHAP explainer if available
            if hasattr(self.explainer, 'shap_explainer'):
                # Filter warnings during reconfiguration
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)

                    # Set new parameters for the explainer
                    self.explainer.shap_explainer.nsamples = n_samples
                    self.explainer.shap_explainer.l1_reg = l1_reg
                    self.explainer.shap_explainer.feature_perturbation = feature_perturbation

                print(f"SHAP explainer configured with: nsamples={n_samples}, l1_reg={l1_reg}")
                return True
            else:
                print("Warning: SHAP explainer not directly accessible for configuration")
                return False
        except Exception as e:
            print(f"Error configuring SHAP explainer: {e}")
            return False

    def _get_token_importance_ranking(self, text: str, shap_values: np.ndarray,
                                      absolute: bool = True) -> List[Tuple[int, float]]:
        """
        Get token importance ranking from SHAP values

        Args:
            text: Original text
            shap_values: SHAP values array
            absolute: Whether to use absolute values for ranking

        Returns:
            List of (token_index, importance_score) tuples sorted by importance
        """
        tokens = self.explainer.preprocess_text(text)

        # Get importance scores for valid tokens only
        token_importance = []
        for i, token in enumerate(tokens):
            if i < len(shap_values[0]):
                importance = abs(shap_values[0][i]) if absolute else shap_values[0][i]
                token_importance.append((i, importance))

        # Sort by importance (descending)
        token_importance.sort(key=lambda x: x[1], reverse=True)
        return token_importance

    def _create_masked_sequence(self, text: str, mask_indices: List[int],
                                mask_strategy: str = 'zero') -> torch.Tensor:
        """
        Create a masked version of the text sequence

        Args:
            text: Original text
            mask_indices: Indices of tokens to mask
            mask_strategy: How to mask ('zero', 'random', 'unk')

        Returns:
            Masked sequence tensor
        """
        sequence = self.explainer.text_to_sequence(text).clone()

        # Filter out indices that are out of bounds
        valid_mask_indices = [idx for idx in mask_indices if idx < len(sequence)]

        if mask_strategy == 'zero':
            # Replace with padding token (0)
            sequence[valid_mask_indices] = 0
        elif mask_strategy == 'random':
            # Replace with random tokens from vocabulary
            vocab_size = len(self.explainer.word_to_idx)
            random_tokens = torch.randint(1, vocab_size, (len(valid_mask_indices),))
            sequence[valid_mask_indices] = random_tokens
        elif mask_strategy == 'unk':
            # Replace with unknown token if available
            unk_idx = self.explainer.word_to_idx.get('<UNK>', 0)
            sequence[valid_mask_indices] = unk_idx

        return sequence

    def compute_auc_deletion(self, text: str, shap_values: np.ndarray,
                             steps: int = 20, mask_strategy: str = 'zero') -> float:
        """
        Compute AUC-Del: Area under deletion curve
        Lower values indicate better explanations

        Args:
            text: Original text
            shap_values: SHAP values for the text
            steps: Number of deletion steps
            mask_strategy: How to mask tokens ('zero', 'random', 'unk')

        Returns:
            AUC-Del score (lower is better)
        """
        # Get original prediction
        original_seq = self.explainer.text_to_sequence(text).unsqueeze(0).to(self.device)
        with torch.no_grad():
            original_pred = self.model.predict(original_seq).item()

        # Get token importance ranking
        token_ranking = self._get_token_importance_ranking(text, shap_values, absolute=True)

        if not token_ranking:
            return 0.0

        # Compute predictions for progressive deletion
        predictions = [original_pred]
        x_values = [0]

        tokens_to_remove = []
        step_size = max(1, len(token_ranking) // steps)

        for i in range(0, len(token_ranking), step_size):
            # Add next batch of most important tokens to removal list
            batch_end = min(i + step_size, len(token_ranking))
            tokens_to_remove.extend([idx for idx, _ in token_ranking[i:batch_end]])

            # Create masked sequence
            masked_seq = self._create_masked_sequence(text, tokens_to_remove, mask_strategy)
            masked_seq = masked_seq.unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                masked_pred = self.model.predict(masked_seq).item()

            predictions.append(masked_pred)
            x_values.append(len(tokens_to_remove))

        # Normalize x-values to [0, 1]
        if len(token_ranking) > 0:
            x_normalized = [x / len(token_ranking) for x in x_values]
        else:
            x_normalized = x_values

        # Calculate AUC (lower is better for deletion)
        auc_del = auc(x_normalized, predictions)
        return auc_del

    def compute_auc_insertion(self, text: str, shap_values: np.ndarray,
                              steps: int = 20) -> float:
        """
        Compute AUC-Ins: Area under insertion curve
        Higher values indicate better explanations

        Args:
            text: Original text
            shap_values: SHAP values for the text
            steps: Number of insertion steps

        Returns:
            AUC-Ins score (higher is better)
        """
        # Start with completely masked sequence
        tokens = self.explainer.preprocess_text(text)
        if not tokens:
            return 0.0

        all_masked_seq = self._create_masked_sequence(text, list(range(len(tokens))), 'zero')
        all_masked_seq = all_masked_seq.unsqueeze(0).to(self.device)

        with torch.no_grad():
            baseline_pred = self.model.predict(all_masked_seq).item()

        # Get token importance ranking
        token_ranking = self._get_token_importance_ranking(text, shap_values, absolute=True)

        # Compute predictions for progressive insertion
        predictions = [baseline_pred]
        x_values = [0]

        tokens_to_keep = []
        step_size = max(1, len(token_ranking) // steps)

        for i in range(0, len(token_ranking), step_size):
            # Add next batch of most important tokens
            batch_end = min(i + step_size, len(token_ranking))
            tokens_to_keep.extend([idx for idx, _ in token_ranking[i:batch_end]])

            # Create sequence with only these tokens unmasked
            all_indices = set(range(len(tokens)))
            mask_indices = list(all_indices - set(tokens_to_keep))

            partially_masked_seq = self._create_masked_sequence(text, mask_indices, 'zero')
            partially_masked_seq = partially_masked_seq.unsqueeze(0).to(self.device)

            # Get prediction
            with torch.no_grad():
                pred = self.model.predict(partially_masked_seq).item()

            predictions.append(pred)
            x_values.append(len(tokens_to_keep))

        # Normalize x-values to [0, 1]
        if len(token_ranking) > 0:
            x_normalized = [x / len(token_ranking) for x in x_values]
        else:
            x_normalized = x_values

        # Calculate AUC (higher is better for insertion)
        auc_ins = auc(x_normalized, predictions)
        return auc_ins

    def compute_comprehensiveness(self, text: str, shap_values: np.ndarray,
                                  k: int = 5) -> float:
        """
        Compute comprehensiveness: prediction change when removing top-k features
        Higher values indicate better explanations

        Args:
            text: Original text
            shap_values: SHAP values for the text
            k: Number of top features to remove

        Returns:
            Comprehensiveness score (higher is better)
        """
        # Get original prediction
        original_seq = self.explainer.text_to_sequence(text).unsqueeze(0).to(self.device)
        with torch.no_grad():
            original_pred = self.model.predict(original_seq).item()

        # Get top-k most important tokens
        token_ranking = self._get_token_importance_ranking(text, shap_values, absolute=True)
        top_k_indices = [idx for idx, _ in token_ranking[:k]]

        if not top_k_indices:
            return 0.0

        # Create sequence with top-k tokens removed
        masked_seq = self._create_masked_sequence(text, top_k_indices, 'zero')
        masked_seq = masked_seq.unsqueeze(0).to(self.device)

        # Get prediction without top-k features
        with torch.no_grad():
            masked_pred = self.model.predict(masked_seq).item()

        # Comprehensiveness is the absolute difference
        comprehensiveness = abs(original_pred - masked_pred)
        return comprehensiveness

    def compute_jaccard_stability(self, text: str, num_perturbations: int = 10,
                                  k: int = 5, perturbation_prob: float = 0.1) -> float:
        """
        Compute Jaccard stability: similarity of top-k features across perturbed inputs
        Higher values indicate more stable explanations

        Args:
            text: Original text
            num_perturbations: Number of perturbed versions to generate
            k: Number of top features to compare
            perturbation_prob: Probability of perturbing each token

        Returns:
            Average Jaccard similarity (higher is better)
        """
        # Get original explanation
        original_shap = self.explainer.explain_prediction(text)
        original_ranking = self._get_token_importance_ranking(text, original_shap, absolute=True)
        original_top_k = set([idx for idx, _ in original_ranking[:k]])

        jaccard_scores = []
        tokens = self.explainer.preprocess_text(text)

        for _ in range(num_perturbations):
            # Create perturbed version by randomly replacing some tokens
            perturbed_tokens = tokens.copy()

            for i in range(len(perturbed_tokens)):
                if random.random() < perturbation_prob:
                    # Replace with a random token from vocabulary
                    vocab_words = list(self.explainer.word_to_idx.keys())
                    perturbed_tokens[i] = random.choice(vocab_words)

            # Reconstruct text (simple approach)
            perturbed_text = ' '.join(perturbed_tokens)

            try:
                # Get explanation for perturbed text
                perturbed_shap = self.explainer.explain_prediction(perturbed_text)
                perturbed_ranking = self._get_token_importance_ranking(perturbed_text, perturbed_shap, absolute=True)
                perturbed_top_k = set([idx for idx, _ in perturbed_ranking[:k]])

                # Calculate Jaccard similarity
                if len(original_top_k) == 0 and len(perturbed_top_k) == 0:
                    jaccard = 1.0
                elif len(original_top_k) == 0 or len(perturbed_top_k) == 0:
                    jaccard = 0.0
                else:
                    intersection = len(original_top_k.intersection(perturbed_top_k))
                    union = len(original_top_k.union(perturbed_top_k))
                    jaccard = intersection / union if union > 0 else 0.0

                jaccard_scores.append(jaccard)

            except Exception as e:
                print(f"Error processing perturbed text: {e}")
                continue

        return np.mean(jaccard_scores) if jaccard_scores else 0.0

    def compute_rank_correlation(self, text: str, num_perturbations: int = 10,
                                 perturbation_prob: float = 0.05) -> float:
        """
        Compute rank correlation: Spearman's ρ between explanation weights for perturbed samples
        Higher values indicate more stable explanations

        Args:
            text: Original text
            num_perturbations: Number of perturbed versions to generate
            perturbation_prob: Probability of perturbing each token

        Returns:
            Average Spearman correlation (higher is better)
        """
        # Get original explanation
        original_shap = self.explainer.explain_prediction(text)
        original_values = original_shap[0]

        correlations = []
        tokens = self.explainer.preprocess_text(text)

        for _ in range(num_perturbations):
            # Create slightly perturbed version
            perturbed_tokens = tokens.copy()

            for i in range(len(perturbed_tokens)):
                if random.random() < perturbation_prob:
                    vocab_words = list(self.explainer.word_to_idx.keys())
                    perturbed_tokens[i] = random.choice(vocab_words)

            perturbed_text = ' '.join(perturbed_tokens)

            try:
                # Get explanation for perturbed text
                perturbed_shap = self.explainer.explain_prediction(perturbed_text)
                perturbed_values = perturbed_shap[0]

                # Ensure same length for comparison
                min_len = min(len(original_values), len(perturbed_values))
                if min_len > 1:
                    correlation, _ = spearmanr(
                        original_values[:min_len],
                        perturbed_values[:min_len]
                    )

                    if not np.isnan(correlation):
                        correlations.append(correlation)

            except Exception as e:
                print(f"Error computing correlation: {e}")
                continue

        return np.mean(correlations) if correlations else 0.0

    def evaluate_explanation_quality(self, text: str, shap_values: np.ndarray,
                                     verbose: bool = True) -> Dict[str, float]:
        """
        Compute all explanation quality metrics for a given text and its SHAP values

        Args:
            text: Input text
            shap_values: SHAP values for the text
            verbose: Whether to print detailed results

        Returns:
            Dictionary containing all metric scores
        """
        print(f"Evaluating explanation quality for text: '{text[:50]}...'" if len(
            text) > 50 else f"Evaluating explanation quality for text: '{text}'")

        metrics = {}

        # AUC-Del (lower is better)
        print("Computing AUC-Del...")
        metrics['auc_deletion'] = self.compute_auc_deletion(text, shap_values)

        # AUC-Ins (higher is better)
        print("Computing AUC-Ins...")
        metrics['auc_insertion'] = self.compute_auc_insertion(text, shap_values)

        # Comprehensiveness (higher is better)
        print("Computing Comprehensiveness...")
        metrics['comprehensiveness'] = self.compute_comprehensiveness(text, shap_values)

        # Jaccard Stability (higher is better)
        print("Computing Jaccard Stability...")
        metrics['jaccard_stability'] = self.compute_jaccard_stability(text)

        # Rank Correlation (higher is better)
        print("Computing Rank Correlation...")
        metrics['rank_correlation'] = self.compute_rank_correlation(text)

        if verbose:
            print("\n" + "=" * 50)
            print("EXPLANATION QUALITY METRICS")
            print("=" * 50)
            print(f"AUC-Deletion:     {metrics['auc_deletion']:.4f} (lower is better)")
            print(f"AUC-Insertion:    {metrics['auc_insertion']:.4f} (higher is better)")
            print(f"Comprehensiveness: {metrics['comprehensiveness']:.4f} (higher is better)")
            print(f"Jaccard Stability: {metrics['jaccard_stability']:.4f} (higher is better)")
            print(f"Rank Correlation:  {metrics['rank_correlation']:.4f} (higher is better)")
            print("=" * 50)

        return metrics

    def plot_deletion_insertion_curves(self, text: str, shap_values: np.ndarray,
                                       steps: int = 20, save_path: str = None):
        """
        Plot deletion and insertion curves for visualization

        Args:
            text: Input text
            shap_values: SHAP values for the text
            steps: Number of steps for the curves
            save_path: Path to save the plot
        """
        # Compute deletion curve
        original_seq = self.explainer.text_to_sequence(text).unsqueeze(0).to(self.device)
        with torch.no_grad():
            original_pred = self.model.predict(original_seq).item()

        token_ranking = self._get_token_importance_ranking(text, shap_values, absolute=True)

        # Deletion curve
        del_predictions = [original_pred]
        del_x = [0]
        tokens_to_remove = []
        step_size = max(1, len(token_ranking) // steps)

        for i in range(0, len(token_ranking), step_size):
            batch_end = min(i + step_size, len(token_ranking))
            tokens_to_remove.extend([idx for idx, _ in token_ranking[i:batch_end]])

            masked_seq = self._create_masked_sequence(text, tokens_to_remove, 'zero')
            masked_seq = masked_seq.unsqueeze(0).to(self.device)

            with torch.no_grad():
                masked_pred = self.model.predict(masked_seq).item()

            del_predictions.append(masked_pred)
            del_x.append(len(tokens_to_remove))

        # Insertion curve
        tokens = self.explainer.preprocess_text(text)
        all_masked_seq = self._create_masked_sequence(text, list(range(len(tokens))), 'zero')
        all_masked_seq = all_masked_seq.unsqueeze(0).to(self.device)

        with torch.no_grad():
            baseline_pred = self.model.predict(all_masked_seq).item()

        ins_predictions = [baseline_pred]
        ins_x = [0]
        tokens_to_keep = []

        for i in range(0, len(token_ranking), step_size):
            batch_end = min(i + step_size, len(token_ranking))
            tokens_to_keep.extend([idx for idx, _ in token_ranking[i:batch_end]])

            all_indices = set(range(len(tokens)))
            mask_indices = list(all_indices - set(tokens_to_keep))

            partially_masked_seq = self._create_masked_sequence(text, mask_indices, 'zero')
            partially_masked_seq = partially_masked_seq.unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.model.predict(partially_masked_seq).item()

            ins_predictions.append(pred)
            ins_x.append(len(tokens_to_keep))

        # Normalize x-values
        if len(token_ranking) > 0:
            del_x_norm = [x / len(token_ranking) for x in del_x]
            ins_x_norm = [x / len(token_ranking) for x in ins_x]
        else:
            del_x_norm = del_x
            ins_x_norm = ins_x

        # Create plot
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(del_x_norm, del_predictions, 'r-', linewidth=2, label='Deletion Curve')
        plt.xlabel('Fraction of Features Removed')
        plt.ylabel('Prediction Score')
        plt.title('Deletion Curve (Lower AUC = Better)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(ins_x_norm, ins_predictions, 'g-', linewidth=2, label='Insertion Curve')
        plt.xlabel('Fraction of Features Added')
        plt.ylabel('Prediction Score')
        plt.title('Insertion Curve (Higher AUC = Better)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


# Example usage function
def demonstrate_quality_metrics(explainer, test_text: str):
    """
    Demonstrate how to use the explanation quality metrics

    Args:
        explainer: Initialized SpamCNNExplainer
        test_text: Text to analyze
    """
    print("Initializing quality metrics calculator...")
    quality_evaluator = ExplanationQualityMetrics(explainer)

    # Configure SHAP explainer for better numerical stability
    print("Configuring SHAP explainer for numerical stability...")
    quality_evaluator.configure_shap_for_stability(l1_reg='num_features(10)', n_samples=100)

    print("Generating SHAP explanation...")
    shap_values = explainer.explain_prediction(test_text)

    print("Computing explanation quality metrics...")
    metrics = quality_evaluator.evaluate_explanation_quality(test_text, shap_values)

    print("Plotting deletion and insertion curves...")
    quality_evaluator.plot_deletion_insertion_curves(test_text, shap_values)

    return metrics
