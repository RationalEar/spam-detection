import random
from typing import List, Dict, Tuple, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import auc
from transformers import BertTokenizer


class BertExplanationMetrics:
    """
    Explanation quality metrics calculator for BERT models using LayerIntegratedGradients
    and attention heads.
    """
    
    def __init__(self, model, tokenizer, device='cpu'):
        """
        Initialize explanation quality metrics calculator for BERT

        Args:
            model: SpamBERT model instance
            tokenizer: BERT tokenizer
            device: Device to run computations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()


    def _tokenize_text(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Tokenize text for BERT input
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing input_ids, attention_mask, and token_type_ids
        """
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device),
            'token_type_ids': encoded.get('token_type_ids', None)
        }


    def _get_model_prediction(self, input_ids: torch.Tensor, 
                             attention_mask: torch.Tensor = None,
                             token_type_ids: torch.Tensor = None) -> float:
        """
        Get model prediction for given inputs
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            
        Returns:
            Model prediction probability
        """
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            if isinstance(output, tuple):
                return output[0].cpu().item()
            return output.cpu().item()


    def _get_integrated_gradients(self, text: str, n_steps: int = 50) -> Tuple[torch.Tensor, float]:
        """
        Compute integrated gradients for the given text
        
        Args:
            text: Input text
            n_steps: Number of steps for integrated gradients
            
        Returns:
            Tuple of (attributions, convergence_delta)
        """
        encoded = self._tokenize_text(text)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        
        # Compute integrated gradients
        attributions, delta = self.model.compute_integrated_gradients(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            n_steps=n_steps
        )
        
        return attributions.squeeze(0), delta.item()


    def _get_attention_weights(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Get attention weights from the model
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of attention weights by layer
        """
        encoded = self._tokenize_text(text)
        
        with torch.no_grad():
            _, attention_data = self.model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                token_type_ids=encoded['token_type_ids'],
                return_attentions=True
            )
        
        return attention_data if attention_data else {}


    def _get_token_importance_ranking(self, attributions: torch.Tensor, 
                                    attention_mask: torch.Tensor = None,
                                    absolute: bool = True) -> List[Tuple[int, float]]:
        """
        Get token importance ranking from attribution scores
        
        Args:
            attributions: Attribution scores for each token
            attention_mask: Mask to identify valid tokens
            absolute: Whether to use absolute values for ranking
            
        Returns:
            List of (token_index, importance_score) tuples sorted by importance
        """
        # Sum attributions across embedding dimensions if needed
        if len(attributions.shape) > 1:
            token_scores = attributions.sum(dim=-1)
        else:
            token_scores = attributions
            
        # Apply attention mask if provided
        if attention_mask is not None:
            token_scores = token_scores * attention_mask.squeeze(0).float()
        
        # Get importance scores
        importance_scores = torch.abs(token_scores) if absolute else token_scores
        
        # Create ranking
        token_importance = []
        for i, score in enumerate(importance_scores):
            if attention_mask is None or attention_mask.squeeze(0)[i] == 1:
                token_importance.append((i, score.item()))
        
        # Sort by importance (descending)
        token_importance.sort(key=lambda x: x[1], reverse=True)
        return token_importance


    def _create_masked_input(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                           mask_indices: List[int], mask_strategy: str = 'mask') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masked version of input by masking specified token positions
        
        Args:
            input_ids: Original input token IDs
            attention_mask: Original attention mask
            mask_indices: Indices of tokens to mask
            mask_strategy: How to mask ('mask', 'pad', 'random')
            
        Returns:
            Tuple of (masked_input_ids, masked_attention_mask)
        """
        masked_input_ids = input_ids.clone()
        masked_attention_mask = attention_mask.clone()
        
        for idx in mask_indices:
            if idx < input_ids.size(1):  # Check bounds
                if mask_strategy == 'mask':
                    # Use BERT's [MASK] token
                    masked_input_ids[0, idx] = self.tokenizer.mask_token_id
                elif mask_strategy == 'pad':
                    # Use padding token and set attention to 0
                    masked_input_ids[0, idx] = self.tokenizer.pad_token_id
                    masked_attention_mask[0, idx] = 0
                elif mask_strategy == 'random':
                    # Replace with random token from vocabulary
                    vocab_size = self.tokenizer.vocab_size
                    random_token = torch.randint(1, vocab_size, (1,)).item()
                    masked_input_ids[0, idx] = random_token
        
        return masked_input_ids, masked_attention_mask


    def compute_auc_deletion(self, text: str, method: str = 'integrated_gradients',
                           steps: int = 20, mask_strategy: str = 'mask') -> float:
        """
        Compute AUC-Del: Area under deletion curve
        Lower values indicate better explanations
        
        Args:
            text: Original text
            method: Attribution method ('integrated_gradients' or 'attention')
            steps: Number of deletion steps
            mask_strategy: How to mask tokens ('mask', 'pad', 'random')
            
        Returns:
            AUC-Del score (lower is better)
        """
        # Get original prediction
        encoded = self._tokenize_text(text)
        original_pred = self._get_model_prediction(
            encoded['input_ids'], 
            encoded['attention_mask'],
            encoded['token_type_ids']
        )
        
        # Get attribution scores
        if method == 'integrated_gradients':
            attributions, _ = self._get_integrated_gradients(text)
            token_ranking = self._get_token_importance_ranking(
                attributions, encoded['attention_mask'], absolute=True
            )
        elif method == 'attention':
            attention_weights = self._get_attention_weights(text)
            if not attention_weights:
                return 0.0
            # Use average attention from the last layer
            last_layer_key = list(attention_weights.keys())[-1]
            avg_attention = attention_weights[last_layer_key].mean(dim=1).squeeze(0)
            token_ranking = self._get_token_importance_ranking(
                avg_attention, encoded['attention_mask'], absolute=True
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
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
            
            # Create masked input
            masked_input_ids, masked_attention_mask = self._create_masked_input(
                encoded['input_ids'], encoded['attention_mask'], 
                tokens_to_remove, mask_strategy
            )
            
            # Get prediction
            masked_pred = self._get_model_prediction(
                masked_input_ids, masked_attention_mask, encoded['token_type_ids']
            )
            
            predictions.append(masked_pred)
            x_values.append(len(tokens_to_remove))
        
        # Normalize x-values to [0, 1]
        max_tokens = len(token_ranking)
        if max_tokens > 0:
            x_normalized = [x / max_tokens for x in x_values]
        else:
            x_normalized = x_values
        
        # Calculate AUC (lower is better for deletion)
        if len(x_normalized) > 1 and len(predictions) > 1:
            auc_del = auc(x_normalized, predictions)
        else:
            auc_del = 0.0
            
        return auc_del


    def compute_auc_insertion(self, text: str, method: str = 'integrated_gradients',
                            steps: int = 20) -> float:
        """
        Compute AUC-Ins: Area under insertion curve
        Higher values indicate better explanations
        
        Args:
            text: Original text
            method: Attribution method ('integrated_gradients' or 'attention')
            steps: Number of insertion steps
            
        Returns:
            AUC-Ins score (higher is better)
        """
        encoded = self._tokenize_text(text)
        
        # Start with completely masked sequence
        valid_tokens = []
        for i, token_id in enumerate(encoded['input_ids'].squeeze(0)):
            if encoded['attention_mask'].squeeze(0)[i] == 1 and token_id not in [
                self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id
            ]:
                valid_tokens.append(i)
        
        all_masked_input_ids, all_masked_attention_mask = self._create_masked_input(
            encoded['input_ids'], encoded['attention_mask'], valid_tokens, 'mask'
        )
        
        baseline_pred = self._get_model_prediction(
            all_masked_input_ids, all_masked_attention_mask, encoded['token_type_ids']
        )
        
        # Get attribution scores
        if method == 'integrated_gradients':
            attributions, _ = self._get_integrated_gradients(text)
            token_ranking = self._get_token_importance_ranking(
                attributions, encoded['attention_mask'], absolute=True
            )
        elif method == 'attention':
            attention_weights = self._get_attention_weights(text)
            if not attention_weights:
                return 0.0
            last_layer_key = list(attention_weights.keys())[-1]
            avg_attention = attention_weights[last_layer_key].mean(dim=1).squeeze(0)
            token_ranking = self._get_token_importance_ranking(
                avg_attention, encoded['attention_mask'], absolute=True
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Filter out special tokens from ranking
        filtered_ranking = [
            (idx, score) for idx, score in token_ranking 
            if idx in valid_tokens
        ]
        
        if not filtered_ranking:
            return 0.0
        
        # Compute predictions for progressive insertion
        predictions = [baseline_pred]
        x_values = [0]
        
        tokens_to_keep = []
        step_size = max(1, len(filtered_ranking) // steps)
        
        for i in range(0, len(filtered_ranking), step_size):
            # Add next batch of most important tokens
            batch_end = min(i + step_size, len(filtered_ranking))
            tokens_to_keep.extend([idx for idx, _ in filtered_ranking[i:batch_end]])
            
            # Create sequence with only these tokens unmasked
            tokens_to_mask = [idx for idx in valid_tokens if idx not in tokens_to_keep]
            
            partially_masked_input_ids, partially_masked_attention_mask = self._create_masked_input(
                encoded['input_ids'], encoded['attention_mask'], tokens_to_mask, 'mask'
            )
            
            # Get prediction
            pred = self._get_model_prediction(
                partially_masked_input_ids, partially_masked_attention_mask, encoded['token_type_ids']
            )
            
            predictions.append(pred)
            x_values.append(len(tokens_to_keep))
        
        # Normalize x-values to [0, 1]
        max_tokens = len(filtered_ranking)
        if max_tokens > 0:
            x_normalized = [x / max_tokens for x in x_values]
        else:
            x_normalized = x_values
        
        # Calculate AUC (higher is better for insertion)
        if len(x_normalized) > 1 and len(predictions) > 1:
            auc_ins = auc(x_normalized, predictions)
        else:
            auc_ins = 0.0
            
        return auc_ins


    def compute_comprehensiveness(self, text: str, method: str = 'integrated_gradients',
                                k: int = 5) -> float:
        """
        Compute comprehensiveness: prediction change when removing top-k features
        Higher values indicate better explanations
        
        Args:
            text: Original text
            method: Attribution method ('integrated_gradients' or 'attention')
            k: Number of top features to remove
            
        Returns:
            Comprehensiveness score (higher is better)
        """
        # Get original prediction
        encoded = self._tokenize_text(text)
        original_pred = self._get_model_prediction(
            encoded['input_ids'], encoded['attention_mask'], encoded['token_type_ids']
        )
        
        # Get attribution scores
        if method == 'integrated_gradients':
            attributions, _ = self._get_integrated_gradients(text)
            token_ranking = self._get_token_importance_ranking(
                attributions, encoded['attention_mask'], absolute=True
            )
        elif method == 'attention':
            attention_weights = self._get_attention_weights(text)
            if not attention_weights:
                return 0.0
            last_layer_key = list(attention_weights.keys())[-1]
            avg_attention = attention_weights[last_layer_key].mean(dim=1).squeeze(0)
            token_ranking = self._get_token_importance_ranking(
                avg_attention, encoded['attention_mask'], absolute=True
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get top-k most important tokens
        top_k_indices = [idx for idx, _ in token_ranking[:k]]
        
        if not top_k_indices:
            return 0.0
        
        # Create sequence with top-k tokens masked
        masked_input_ids, masked_attention_mask = self._create_masked_input(
            encoded['input_ids'], encoded['attention_mask'], top_k_indices, 'mask'
        )
        
        # Get prediction without top-k features
        masked_pred = self._get_model_prediction(
            masked_input_ids, masked_attention_mask, encoded['token_type_ids']
        )
        
        # Comprehensiveness is the absolute difference
        comprehensiveness = abs(original_pred - masked_pred)
        return comprehensiveness


    def compute_jaccard_stability(self, text: str, method: str = 'integrated_gradients',
                                 num_perturbations: int = 10, k: int = 5, 
                                 perturbation_prob: float = 0.1) -> float:
        """
        Compute Jaccard stability: similarity of top-k features across perturbed inputs
        Higher values indicate more stable explanations
        
        Args:
            text: Original text
            method: Attribution method ('integrated_gradients' or 'attention')
            num_perturbations: Number of perturbed versions to generate
            k: Number of top features to compare
            perturbation_prob: Probability of perturbing each token
            
        Returns:
            Average Jaccard similarity (higher is better)
        """
        # Get original explanation
        encoded = self._tokenize_text(text)
        
        if method == 'integrated_gradients':
            attributions, _ = self._get_integrated_gradients(text)
            original_ranking = self._get_token_importance_ranking(
                attributions, encoded['attention_mask'], absolute=True
            )
        elif method == 'attention':
            attention_weights = self._get_attention_weights(text)
            if not attention_weights:
                return 0.0
            last_layer_key = list(attention_weights.keys())[-1]
            avg_attention = attention_weights[last_layer_key].mean(dim=1).squeeze(0)
            original_ranking = self._get_token_importance_ranking(
                avg_attention, encoded['attention_mask'], absolute=True
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        original_top_k = set([idx for idx, _ in original_ranking[:k]])
        
        jaccard_scores = []
        
        # Tokenize text to get individual tokens for perturbation
        tokens = self.tokenizer.tokenize(text)
        
        for _ in range(num_perturbations):
            # Create perturbed version by randomly replacing some tokens
            perturbed_tokens = tokens.copy()
            
            for i in range(len(perturbed_tokens)):
                if random.random() < perturbation_prob:
                    # Replace with a random token from vocabulary
                    vocab_size = min(10000, self.tokenizer.vocab_size)  # Limit for efficiency
                    random_token_id = random.randint(1000, vocab_size - 1)
                    random_token = self.tokenizer.convert_ids_to_tokens([random_token_id])[0]
                    perturbed_tokens[i] = random_token
            
            # Reconstruct text
            perturbed_text = self.tokenizer.convert_tokens_to_string(perturbed_tokens)
            
            try:
                # Get explanation for perturbed text
                if method == 'integrated_gradients':
                    perturbed_attributions, _ = self._get_integrated_gradients(perturbed_text)
                    perturbed_encoded = self._tokenize_text(perturbed_text)
                    perturbed_ranking = self._get_token_importance_ranking(
                        perturbed_attributions, perturbed_encoded['attention_mask'], absolute=True
                    )
                elif method == 'attention':
                    perturbed_attention_weights = self._get_attention_weights(perturbed_text)
                    if not perturbed_attention_weights:
                        continue
                    last_layer_key = list(perturbed_attention_weights.keys())[-1]
                    perturbed_avg_attention = perturbed_attention_weights[last_layer_key].mean(dim=1).squeeze(0)
                    perturbed_encoded = self._tokenize_text(perturbed_text)
                    perturbed_ranking = self._get_token_importance_ranking(
                        perturbed_avg_attention, perturbed_encoded['attention_mask'], absolute=True
                    )
                
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


    def evaluate_explanation_quality(self, text: str, method: str = 'integrated_gradients',
                                   verbose: bool = True) -> Dict[str, float]:
        """
        Compute all explanation quality metrics for a given text
        
        Args:
            text: Input text
            method: Attribution method ('integrated_gradients' or 'attention')
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary containing all metric scores
        """
        if verbose:
            print(f"Evaluating explanation quality for text: '{text[:50]}...'" if len(text) > 50 else f"Evaluating explanation quality for text: '{text}'")
            print(f"Using method: {method}")
        
        metrics = {}
        
        try:
            # AUC-Del (lower is better)
            if verbose:
                print("Computing AUC-Del...")
            metrics['auc_deletion'] = self.compute_auc_deletion(text, method)
            
            # AUC-Ins (higher is better)
            if verbose:
                print("Computing AUC-Ins...")
            metrics['auc_insertion'] = self.compute_auc_insertion(text, method)
            
            # Comprehensiveness (higher is better)
            if verbose:
                print("Computing Comprehensiveness...")
            metrics['comprehensiveness'] = self.compute_comprehensiveness(text, method)
            
            # Jaccard Stability (higher is better)
            if verbose:
                print("Computing Jaccard Stability...")
            metrics['jaccard_stability'] = self.compute_jaccard_stability(text, method)
            
            if verbose:
                print("\n" + "=" * 50)
                print("EXPLANATION QUALITY METRICS")
                print("=" * 50)
                print(f"Method:           {method}")
                print(f"AUC-Deletion:     {metrics['auc_deletion']:.4f} (lower is better)")
                print(f"AUC-Insertion:    {metrics['auc_insertion']:.4f} (higher is better)")
                print(f"Comprehensiveness: {metrics['comprehensiveness']:.4f} (higher is better)")
                print(f"Jaccard Stability: {metrics['jaccard_stability']:.4f} (higher is better)")
                print("=" * 50)
                
        except Exception as e:
            print(f"Error computing metrics: {e}")
            metrics = {
                'auc_deletion': 0.0,
                'auc_insertion': 0.0,
                'comprehensiveness': 0.0,
                'jaccard_stability': 0.0
            }
        
        return metrics


    def plot_deletion_insertion_curves(self, text: str, method: str = 'integrated_gradients',
                                     steps: int = 20, save_path: str = None):
        """
        Plot deletion and insertion curves for visualization
        
        Args:
            text: Input text
            method: Attribution method to use
            steps: Number of steps for the curves
            save_path: Path to save the plot
        """
        encoded = self._tokenize_text(text)
        original_pred = self._get_model_prediction(
            encoded['input_ids'], encoded['attention_mask'], encoded['token_type_ids']
        )
        
        # Get attribution scores
        if method == 'integrated_gradients':
            attributions, _ = self._get_integrated_gradients(text)
            token_ranking = self._get_token_importance_ranking(
                attributions, encoded['attention_mask'], absolute=True
            )
        elif method == 'attention':
            attention_weights = self._get_attention_weights(text)
            if not attention_weights:
                print("No attention weights available")
                return
            last_layer_key = list(attention_weights.keys())[-1]
            avg_attention = attention_weights[last_layer_key].mean(dim=1).squeeze(0)
            token_ranking = self._get_token_importance_ranking(
                avg_attention, encoded['attention_mask'], absolute=True
            )
        
        # Deletion curve
        del_predictions = [original_pred]
        del_x = [0]
        tokens_to_remove = []
        step_size = max(1, len(token_ranking) // steps)
        
        for i in range(0, len(token_ranking), step_size):
            batch_end = min(i + step_size, len(token_ranking))
            tokens_to_remove.extend([idx for idx, _ in token_ranking[i:batch_end]])
            
            masked_input_ids, masked_attention_mask = self._create_masked_input(
                encoded['input_ids'], encoded['attention_mask'], tokens_to_remove, 'mask'
            )
            
            masked_pred = self._get_model_prediction(
                masked_input_ids, masked_attention_mask, encoded['token_type_ids']
            )
            
            del_predictions.append(masked_pred)
            del_x.append(len(tokens_to_remove))
        
        # Insertion curve
        valid_tokens = []
        for i, token_id in enumerate(encoded['input_ids'].squeeze(0)):
            if encoded['attention_mask'].squeeze(0)[i] == 1 and token_id not in [
                self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id
            ]:
                valid_tokens.append(i)
        
        all_masked_input_ids, all_masked_attention_mask = self._create_masked_input(
            encoded['input_ids'], encoded['attention_mask'], valid_tokens, 'mask'
        )
        
        baseline_pred = self._get_model_prediction(
            all_masked_input_ids, all_masked_attention_mask, encoded['token_type_ids']
        )
        
        ins_predictions = [baseline_pred]
        ins_x = [0]
        tokens_to_keep = []
        
        filtered_ranking = [(idx, score) for idx, score in token_ranking if idx in valid_tokens]
        
        for i in range(0, len(filtered_ranking), step_size):
            batch_end = min(i + step_size, len(filtered_ranking))
            tokens_to_keep.extend([idx for idx, _ in filtered_ranking[i:batch_end]])
            
            tokens_to_mask = [idx for idx in valid_tokens if idx not in tokens_to_keep]
            
            partially_masked_input_ids, partially_masked_attention_mask = self._create_masked_input(
                encoded['input_ids'], encoded['attention_mask'], tokens_to_mask, 'mask'
            )
            
            pred = self._get_model_prediction(
                partially_masked_input_ids, partially_masked_attention_mask, encoded['token_type_ids']
            )
            
            ins_predictions.append(pred)
            ins_x.append(len(tokens_to_keep))
        
        # Normalize x-values
        max_del_tokens = len(token_ranking) if len(token_ranking) > 0 else 1
        max_ins_tokens = len(filtered_ranking) if len(filtered_ranking) > 0 else 1
        
        del_x_norm = [x / max_del_tokens for x in del_x]
        ins_x_norm = [x / max_ins_tokens for x in ins_x]
        
        # Create plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(del_x_norm, del_predictions, 'r-', linewidth=2, label='Deletion Curve')
        plt.xlabel('Fraction of Features Removed')
        plt.ylabel('Prediction Score')
        plt.title(f'Deletion Curve - {method}\n(Lower AUC = Better)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(ins_x_norm, ins_predictions, 'g-', linewidth=2, label='Insertion Curve')
        plt.xlabel('Fraction of Features Added')
        plt.ylabel('Prediction Score')
        plt.title(f'Insertion Curve - {method}\n(Higher AUC = Better)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def demonstrate_bert_quality_metrics(model, tokenizer, test_text: str, device: str = 'cpu'):
    """
    Demonstrate how to use the BERT explanation quality metrics
    
    Args:
        model: SpamBERT model instance
        tokenizer: BERT tokenizer
        test_text: Text to analyze
        device: Device to run on
    """
    print("Initializing BERT quality metrics calculator...")
    quality_evaluator = BertExplanationMetrics(model, tokenizer, device)
    
    print("Computing explanation quality metrics using Integrated Gradients...")
    ig_metrics = quality_evaluator.evaluate_explanation_quality(test_text, method='integrated_gradients')
    
    print("\nComputing explanation quality metrics using Attention Weights...")
    attention_metrics = quality_evaluator.evaluate_explanation_quality(test_text, method='attention')
    
    print("\nPlotting deletion and insertion curves...")
    quality_evaluator.plot_deletion_insertion_curves(test_text, method='integrated_gradients')
    quality_evaluator.plot_deletion_insertion_curves(test_text, method='attention')
    
    return {
        'integrated_gradients': ig_metrics,
        'attention': attention_metrics
    }
