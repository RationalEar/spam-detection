import random
from typing import List, Dict, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import auc


class BiLSTMShapMetrics:
    def __init__(self, shap_explainer, device='cpu'):
        """
        Initialize SHAP-based explanation quality metrics calculator for BiLSTM

        Args:
            shap_explainer: BiLSTMShapExplainer instance
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
        tokens = self.explainer.tokenize_text(text)

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
                                mask_strategy: str = 'remove') -> str:
        """
        Create a masked version of the text

        Args:
            text: Original text
            mask_indices: Indices of tokens to mask
            mask_strategy: How to mask ('remove', 'replace')

        Returns:
            Masked text string
        """
        tokens = self.explainer.tokenize_text(text)

        # Filter out indices that are out of bounds
        valid_mask_indices = [idx for idx in mask_indices if idx < len(tokens)]

        if mask_strategy == 'remove':
            # Remove masked tokens
            masked_tokens = [token for i, token in enumerate(tokens) if i not in valid_mask_indices]
        elif mask_strategy == 'replace':
            # Replace with generic token
            masked_tokens = []
            for i, token in enumerate(tokens):
                if i in valid_mask_indices:
                    masked_tokens.append('<MASK>')
                else:
                    masked_tokens.append(token)
        else:
            # Default: remove tokens
            masked_tokens = [token for i, token in enumerate(tokens) if i not in valid_mask_indices]

        return ' '.join(masked_tokens)

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
            mask_strategy: How to mask tokens ('remove', 'replace')

        Returns:
            AUC-Del score (lower is better)
        """
        # Get original prediction
        original_pred = self._get_prediction(text)

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

            # Create masked text
            masked_text = self._create_masked_sequence(text, tokens_to_remove, mask_strategy)
            
            # Handle empty text case
            if not masked_text.strip():
                masked_pred = 0.0
            else:
                masked_pred = self._get_prediction(masked_text)

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
        # Start with empty text
        baseline_pred = self._get_prediction("")

        # Get token importance ranking
        token_ranking = self._get_token_importance_ranking(text, shap_values, absolute=True)
        tokens = self.explainer.tokenize_text(text)

        if not token_ranking:
            return 0.0

        # Compute predictions for progressive insertion
        predictions = [baseline_pred]
        x_values = [0]

        tokens_to_keep = []
        step_size = max(1, len(token_ranking) // steps)

        for i in range(0, len(token_ranking), step_size):
            # Add next batch of most important tokens
            batch_end = min(i + step_size, len(token_ranking))
            tokens_to_keep.extend([idx for idx, _ in token_ranking[i:batch_end]])

            # Create sequence with only these tokens
            kept_tokens = [tokens[idx] for idx in sorted(tokens_to_keep) if idx < len(tokens)]
            partial_text = ' '.join(kept_tokens)

            # Get prediction
            if not partial_text.strip():
                pred = 0.0
            else:
                pred = self._get_prediction(partial_text)

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
        original_pred = self._get_prediction(text)

        # Get top-k most important tokens
        token_ranking = self._get_token_importance_ranking(text, shap_values, absolute=True)
        top_k_indices = [idx for idx, _ in token_ranking[:k]]

        if not top_k_indices:
            return 0.0

        # Create text with top-k tokens removed
        masked_text = self._create_masked_sequence(text, top_k_indices, 'remove')
        
        # Get prediction without top-k features
        if not masked_text.strip():
            masked_pred = 0.0
        else:
            masked_pred = self._get_prediction(masked_text)

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
        tokens = self.explainer.tokenize_text(text)

        for _ in range(num_perturbations):
            # Create perturbed version by randomly replacing some tokens
            perturbed_tokens = tokens.copy()

            for i in range(len(perturbed_tokens)):
                if random.random() < perturbation_prob:
                    # Replace with a random token from vocabulary
                    vocab_words = [word for word in self.explainer.word_to_idx.keys() 
                                 if word not in ['<PAD>', '<UNK>']]
                    if vocab_words:
                        perturbed_tokens[i] = random.choice(vocab_words)

            # Reconstruct text
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
        Compute rank correlation: Spearman's ρ between SHAP values for perturbed samples
        Higher values indicate more stable explanations

        Args:
            text: Original text
            num_perturbations: Number of perturbed versions to generate
            perturbation_prob: Probability of perturbing each token

        Returns:
            Average Spearman correlation (higher is better)
        """
        # Get original SHAP values
        original_shap = self.explainer.explain_prediction(text)
        tokens = self.explainer.tokenize_text(text)
        
        correlations = []

        for _ in range(num_perturbations):
            # Create perturbed version
            perturbed_tokens = tokens.copy()

            for i in range(len(perturbed_tokens)):
                if random.random() < perturbation_prob:
                    vocab_words = [word for word in self.explainer.word_to_idx.keys() 
                                 if word not in ['<PAD>', '<UNK>']]
                    if vocab_words:
                        perturbed_tokens[i] = random.choice(vocab_words)

            perturbed_text = ' '.join(perturbed_tokens)

            try:
                # Get SHAP values for perturbed text
                perturbed_shap = self.explainer.explain_prediction(perturbed_text)
                
                # Calculate correlation for overlapping positions
                min_len = min(len(original_shap[0]), len(perturbed_shap[0]), len(tokens))
                if min_len > 1:
                    corr, _ = spearmanr(original_shap[0][:min_len], 
                                      perturbed_shap[0][:min_len])
                    if not np.isnan(corr):
                        correlations.append(corr)

            except Exception as e:
                print(f"Error processing perturbed text: {e}")
                continue

        return np.mean(correlations) if correlations else 0.0

    def evaluate_all_metrics(self, text: str, steps: int = 20, k: int = 5,
                           num_perturbations: int = 10, perturbation_prob: float = 0.1,
                           nsamples: int = 100) -> Dict[str, float]:
        """
        Compute all explanation quality metrics for a given text

        Args:
            text: Input text
            steps: Number of steps for AUC calculations
            k: Number of top features for comprehensiveness and Jaccard stability
            num_perturbations: Number of perturbations for stability metrics
            perturbation_prob: Probability of perturbing each token
            nsamples: Number of samples for SHAP explanation

        Returns:
            Dictionary with all metric scores
        """
        print(f"Evaluating SHAP metrics for text: {text[:50]}...")
        
        # Get SHAP explanation
        print("Computing SHAP values...")
        shap_values = self.explainer.explain_prediction(text, nsamples)
        
        results = {}
        
        try:
            results['auc_deletion'] = self.compute_auc_deletion(text, shap_values, steps)
            print(f"AUC-Del: {results['auc_deletion']:.4f}")
        except Exception as e:
            print(f"Error computing AUC-Del: {e}")
            results['auc_deletion'] = 0.0

        try:
            results['auc_insertion'] = self.compute_auc_insertion(text, shap_values, steps)
            print(f"AUC-Ins: {results['auc_insertion']:.4f}")
        except Exception as e:
            print(f"Error computing AUC-Ins: {e}")
            results['auc_insertion'] = 0.0

        try:
            results['comprehensiveness'] = self.compute_comprehensiveness(text, shap_values, k)
            print(f"Comprehensiveness: {results['comprehensiveness']:.4f}")
        except Exception as e:
            print(f"Error computing Comprehensiveness: {e}")
            results['comprehensiveness'] = 0.0

        try:
            results['jaccard_stability'] = self.compute_jaccard_stability(
                text, num_perturbations, k, perturbation_prob)
            print(f"Jaccard Stability: {results['jaccard_stability']:.4f}")
        except Exception as e:
            print(f"Error computing Jaccard Stability: {e}")
            results['jaccard_stability'] = 0.0

        return results

    def compare_with_attention_metrics(self, text: str, attention_metrics_calc,
                                     steps: int = 20, k: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Compare SHAP metrics with attention-based metrics
        
        Args:
            text: Input text
            attention_metrics_calc: BiLSTMAttentionMetrics instance
            steps: Number of steps for AUC calculations
            k: Number of top features
            
        Returns:
            Dictionary comparing both types of metrics
        """
        print("Comparing SHAP vs Attention metrics...")
        
        # Get SHAP metrics
        shap_metrics = self.evaluate_all_metrics(text, steps=steps, k=k)
        
        # Get attention metrics
        attention_metrics = attention_metrics_calc.evaluate_all_metrics(text, steps=steps, k=k)
        
        # Compare
        comparison = {
            'shap': shap_metrics,
            'attention': attention_metrics,
            'differences': {}
        }
        
        for metric in shap_metrics.keys():
            if metric in attention_metrics:
                diff = shap_metrics[metric] - attention_metrics[metric]
                comparison['differences'][metric] = diff
        
        return comparison
