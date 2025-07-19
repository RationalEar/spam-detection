import random
from typing import List, Dict, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import auc


class BertShapMetrics:
    def __init__(self, shap_explainer, device='cpu'):
        """
        Initialize SHAP-based explanation quality metrics calculator for BERT

        Args:
            shap_explainer: BertShapExplainer instance
            device: Device to run computations on
        """
        self.explainer = shap_explainer
        self.model = shap_explainer.model
        self.device = device

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
        # Handle different SHAP value shapes
        if len(shap_values.shape) > 1:
            values = shap_values[0]
        else:
            values = shap_values

        # Get importance scores for tokens
        token_importance = []
        for i in range(len(values)):
            importance = abs(values[i]) if absolute else values[i]
            token_importance.append((i, importance))

        # Sort by importance (descending)
        token_importance.sort(key=lambda x: x[1], reverse=True)
        return token_importance

    def _create_masked_text(self, text: str, mask_indices: List[int],
                           mask_strategy: str = 'remove') -> str:
        """
        Create a masked version of the text

        Args:
            text: Original text
            mask_indices: Indices of tokens to mask
            mask_strategy: How to mask ('remove', 'replace', 'mask_token')

        Returns:
            Masked text string
        """
        # For BERT, we work with words rather than subword tokens for simplicity
        words = text.split()

        # Filter out indices that are out of bounds
        valid_mask_indices = [idx for idx in mask_indices if idx < len(words)]

        if mask_strategy == 'remove':
            # Remove masked words
            masked_words = [word for i, word in enumerate(words) if i not in valid_mask_indices]
        elif mask_strategy == 'replace':
            # Replace with generic token
            masked_words = []
            for i, word in enumerate(words):
                if i in valid_mask_indices:
                    masked_words.append('<UNK>')
                else:
                    masked_words.append(word)
        elif mask_strategy == 'mask_token':
            # Replace with BERT's [MASK] token
            masked_words = []
            for i, word in enumerate(words):
                if i in valid_mask_indices:
                    masked_words.append('[MASK]')
                else:
                    masked_words.append(word)
        else:
            # Default: remove words
            masked_words = [word for i, word in enumerate(words) if i not in valid_mask_indices]

        return ' '.join(masked_words)

    def _get_prediction(self, text: str) -> float:
        """
        Get model prediction for text
        
        Args:
            text: Input text
            
        Returns:
            Prediction probability
        """
        return self.explainer.prediction_function([text])[0]

    def compute_auc_deletion(self, text: str, shap_values: np.ndarray, 
                           steps: int = 20, mask_strategy: str = 'remove') -> float:
        """
        Compute AUC-Del: Area under deletion curve
        Lower values indicate better explanations

        Args:
            text: Original text
            shap_values: SHAP values for the text
            steps: Number of deletion steps
            mask_strategy: How to mask tokens ('remove', 'replace', 'mask_token')

        Returns:
            AUC-Del score (lower is better)
        """
        # Get original prediction
        original_pred = self._get_prediction(text)
        
        # Get token importance ranking
        token_ranking = self._get_token_importance_ranking(text, shap_values)
        
        if not token_ranking:
            return 1.0  # Worst possible score if no tokens
        
        # Compute predictions for progressive deletion
        predictions = [original_pred]
        x_values = [0]
        
        words = text.split()
        tokens_to_remove = []
        step_size = max(1, len(token_ranking) // steps)
        
        for i in range(0, len(token_ranking), step_size):
            # Add tokens to removal list
            end_idx = min(i + step_size, len(token_ranking))
            for j in range(i, end_idx):
                if token_ranking[j][0] < len(words):  # Ensure valid index
                    tokens_to_remove.append(token_ranking[j][0])
            
            # Create masked text
            if tokens_to_remove:
                masked_text = self._create_masked_text(text, tokens_to_remove, mask_strategy)
                if masked_text.strip():  # Only predict if text is not empty
                    masked_pred = self._get_prediction(masked_text)
                else:
                    masked_pred = 0.5  # Neutral prediction for empty text
                
                predictions.append(masked_pred)
                x_values.append(len(tokens_to_remove))
        
        # Normalize x-values to [0, 1]
        max_tokens = len(words)
        if max_tokens > 0:
            x_normalized = [x / max_tokens for x in x_values]
        else:
            x_normalized = [0]
        
        # Calculate AUC (lower is better for deletion)
        if len(x_normalized) > 1 and len(predictions) > 1:
            auc_del = auc(x_normalized, predictions)
        else:
            auc_del = 1.0
            
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
        # Start with empty/masked text and progressively add most important tokens
        words = text.split()
        
        # Get baseline prediction (empty text)
        baseline_pred = self._get_prediction("")
        if np.isnan(baseline_pred):
            baseline_pred = 0.5  # Neutral prediction
        
        # Get token importance ranking
        token_ranking = self._get_token_importance_ranking(text, shap_values)
        
        if not token_ranking:
            return 0.0  # Worst possible score if no tokens
        
        # Filter ranking to valid word indices
        valid_ranking = [(idx, score) for idx, score in token_ranking if idx < len(words)]
        
        if not valid_ranking:
            return 0.0
        
        # Compute predictions for progressive insertion
        predictions = [baseline_pred]
        x_values = [0]
        
        tokens_to_keep = []
        step_size = max(1, len(valid_ranking) // steps)
        
        for i in range(0, len(valid_ranking), step_size):
            # Add tokens to keep list
            end_idx = min(i + step_size, len(valid_ranking))
            for j in range(i, end_idx):
                tokens_to_keep.append(valid_ranking[j][0])
            
            # Create text with only selected tokens
            selected_words = [words[idx] for idx in sorted(tokens_to_keep) if idx < len(words)]
            partial_text = ' '.join(selected_words)
            
            if partial_text.strip():
                partial_pred = self._get_prediction(partial_text)
            else:
                partial_pred = baseline_pred
            
            predictions.append(partial_pred)
            x_values.append(len(tokens_to_keep))
        
        # Normalize x-values to [0, 1]
        max_tokens = len(words)
        if max_tokens > 0:
            x_normalized = [x / max_tokens for x in x_values]
        else:
            x_normalized = [0]
        
        # Calculate AUC (higher is better for insertion)
        if len(x_normalized) > 1 and len(predictions) > 1:
            auc_ins = auc(x_normalized, predictions)
        else:
            auc_ins = 0.0
            
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
        original_pred = self._get_prediction(text)
        
        # Get token importance ranking
        token_ranking = self._get_token_importance_ranking(text, shap_values)
        
        # Get top-k most important tokens
        words = text.split()
        top_k_indices = []
        for idx, _ in token_ranking[:k]:
            if idx < len(words):
                top_k_indices.append(idx)
        
        if not top_k_indices:
            return 0.0  # No valid tokens to remove
        
        # Create text with top-k tokens removed
        masked_text = self._create_masked_text(text, top_k_indices, 'remove')
        
        # Get prediction without top-k features
        if masked_text.strip():
            masked_pred = self._get_prediction(masked_text)
        else:
            masked_pred = 0.5  # Neutral prediction for empty text
        
        # Comprehensiveness is the absolute difference
        comprehensiveness = abs(original_pred - masked_pred)
        return comprehensiveness

    def compute_jaccard_stability(self, text: str, shap_values: np.ndarray,
                                 num_perturbations: int = 10, k: int = 5, 
                                 perturbation_prob: float = 0.1) -> float:
        """
        Compute Jaccard stability: similarity of top-k features across perturbed inputs
        Higher values indicate more stable explanations

        Args:
            text: Original text
            shap_values: SHAP values for the text
            num_perturbations: Number of perturbed versions to generate
            k: Number of top features to compare
            perturbation_prob: Probability of perturbing each word

        Returns:
            Average Jaccard similarity (higher is better)
        """
        # Get original top-k features
        original_ranking = self._get_token_importance_ranking(text, shap_values)
        original_top_k = set([idx for idx, _ in original_ranking[:k]])
        
        if not original_top_k:
            return 0.0
        
        jaccard_scores = []
        words = text.split()
        
        for _ in range(num_perturbations):
            # Create perturbed version by randomly replacing some words
            perturbed_words = []
            for word in words:
                if random.random() < perturbation_prob:
                    # Simple perturbation: replace with random word or remove
                    if random.random() < 0.5:
                        perturbed_words.append('[UNK]')
                    # else: skip word (removal)
                else:
                    perturbed_words.append(word)
            
            perturbed_text = ' '.join(perturbed_words)
            
            if not perturbed_text.strip():
                continue  # Skip empty perturbed text
            
            try:
                # Get SHAP values for perturbed text
                _, perturbed_shap_values = self.explainer.explain_text_simple(perturbed_text)
                
                # Get top-k features for perturbed text
                perturbed_ranking = self._get_token_importance_ranking(perturbed_text, perturbed_shap_values)
                perturbed_top_k = set([idx for idx, _ in perturbed_ranking[:k]])
                
                # Compute Jaccard similarity
                if perturbed_top_k:
                    intersection = len(original_top_k.intersection(perturbed_top_k))
                    union = len(original_top_k.union(perturbed_top_k))
                    jaccard = intersection / union if union > 0 else 0.0
                    jaccard_scores.append(jaccard)
                    
            except Exception as e:
                print(f"Warning: Failed to compute SHAP for perturbed text: {e}")
                continue
        
        return np.mean(jaccard_scores) if jaccard_scores else 0.0

    def evaluate_explanation_quality(self, text: str, shap_values: np.ndarray,
                                   verbose: bool = True) -> Dict[str, float]:
        """
        Compute all explanation quality metrics for a given text

        Args:
            text: Input text
            shap_values: SHAP values for the text
            verbose: Whether to print detailed results

        Returns:
            Dictionary containing all metric scores
        """
        if verbose:
            print(f"Evaluating explanation quality for text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        metrics = {}
        
        try:
            # AUC-Del (lower is better)
            auc_del = self.compute_auc_deletion(text, shap_values)
            metrics['auc_deletion'] = auc_del
            if verbose:
                print(f"AUC-Del: {auc_del:.4f} (lower is better)")
                
        except Exception as e:
            print(f"Error computing AUC-Del: {e}")
            metrics['auc_deletion'] = None
        
        try:
            # AUC-Ins (higher is better)
            auc_ins = self.compute_auc_insertion(text, shap_values)
            metrics['auc_insertion'] = auc_ins
            if verbose:
                print(f"AUC-Ins: {auc_ins:.4f} (higher is better)")
                
        except Exception as e:
            print(f"Error computing AUC-Ins: {e}")
            metrics['auc_insertion'] = None
        
        try:
            # Comprehensiveness (higher is better)
            comprehensiveness = self.compute_comprehensiveness(text, shap_values)
            metrics['comprehensiveness'] = comprehensiveness
            if verbose:
                print(f"Comprehensiveness: {comprehensiveness:.4f} (higher is better)")
                
        except Exception as e:
            print(f"Error computing Comprehensiveness: {e}")
            metrics['comprehensiveness'] = None
        
        try:
            # Jaccard Stability (higher is better)
            jaccard_stability = self.compute_jaccard_stability(text, shap_values)
            metrics['jaccard_stability'] = jaccard_stability
            if verbose:
                print(f"Jaccard Stability: {jaccard_stability:.4f} (higher is better)")
                
        except Exception as e:
            print(f"Error computing Jaccard Stability: {e}")
            metrics['jaccard_stability'] = None
        
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
        original_pred = self._get_prediction(text)
        token_ranking = self._get_token_importance_ranking(text, shap_values)
        words = text.split()
        
        # Deletion curve
        deletion_preds = [original_pred]
        deletion_x = [0]
        tokens_to_remove = []
        step_size = max(1, len(token_ranking) // steps)
        
        for i in range(0, len(token_ranking), step_size):
            end_idx = min(i + step_size, len(token_ranking))
            for j in range(i, end_idx):
                if token_ranking[j][0] < len(words):
                    tokens_to_remove.append(token_ranking[j][0])
            
            if tokens_to_remove:
                masked_text = self._create_masked_text(text, tokens_to_remove, 'remove')
                if masked_text.strip():
                    masked_pred = self._get_prediction(masked_text)
                else:
                    masked_pred = 0.5
                deletion_preds.append(masked_pred)
                deletion_x.append(len(tokens_to_remove) / len(words))
        
        # Insertion curve
        insertion_preds = [self._get_prediction("")]
        insertion_x = [0]
        valid_ranking = [(idx, score) for idx, score in token_ranking if idx < len(words)]
        tokens_to_keep = []
        
        for i in range(0, len(valid_ranking), step_size):
            end_idx = min(i + step_size, len(valid_ranking))
            for j in range(i, end_idx):
                tokens_to_keep.append(valid_ranking[j][0])
            
            selected_words = [words[idx] for idx in sorted(tokens_to_keep) if idx < len(words)]
            partial_text = ' '.join(selected_words)
            
            if partial_text.strip():
                partial_pred = self._get_prediction(partial_text)
            else:
                partial_pred = insertion_preds[0]
            
            insertion_preds.append(partial_pred)
            insertion_x.append(len(tokens_to_keep) / len(words))
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(deletion_x, deletion_preds, 'r-', marker='o', linewidth=2, markersize=4)
        plt.xlabel('Fraction of tokens removed')
        plt.ylabel('Prediction probability')
        plt.title('Deletion Curve (AUC-Del)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(insertion_x, insertion_preds, 'b-', marker='o', linewidth=2, markersize=4)
        plt.xlabel('Fraction of tokens added')
        plt.ylabel('Prediction probability')
        plt.title('Insertion Curve (AUC-Ins)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Deletion/Insertion curves saved to {save_path}")
        
        plt.show()


def demonstrate_bert_shap_quality_metrics(shap_explainer, test_text: str):
    """
    Demonstrate how to use the BERT SHAP explanation quality metrics
    
    Args:
        shap_explainer: BertShapExplainer instance
        test_text: Text to analyze
    """
    print("Initializing BERT SHAP quality metrics calculator...")
    quality_evaluator = BertShapMetrics(shap_explainer)
    
    print("Getting SHAP explanation for the text...")
    tokens, shap_values = shap_explainer.explain_text_simple(test_text)
    
    print("\nComputing explanation quality metrics...")
    metrics = quality_evaluator.evaluate_explanation_quality(test_text, shap_values)
    
    print("\nPlotting deletion and insertion curves...")
    quality_evaluator.plot_deletion_insertion_curves(test_text, shap_values)
    
    return metrics
