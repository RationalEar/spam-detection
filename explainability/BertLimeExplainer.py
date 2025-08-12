import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Callable
from transformers import BertTokenizer
from lime.lime_text import LimeTextExplainer
import re
import random
from sklearn.metrics import auc


class BertLimeExplainer:
    """
    LIME-based explainer for BERT models with explanation quality metrics
    """
    
    def __init__(self, model, tokenizer: BertTokenizer, device='cpu'):
        """
        Initialize BERT LIME explainer
        
        Args:
            model: SpamBERT model instance
            tokenizer: BERT tokenizer
            device: Device to run computations on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Initialize LIME text explainer
        self.lime_explainer = LimeTextExplainer(
            class_names=['Ham', 'Spam'],
            bow=False  # Don't use bag of words, preserve order
        )


    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Prediction function for LIME
        
        Args:
            texts: List of text samples
            
        Returns:
            Prediction probabilities as numpy array
        """
        probabilities = []
        
        for text in texts:
            # Tokenize text
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                if isinstance(output, tuple):
                    prob = output[0].cpu().item()
                else:
                    prob = output.cpu().item()
                
                # Convert to binary classification probabilities
                probabilities.append([1 - prob, prob])  # [prob_ham, prob_spam]
        
        return np.array(probabilities)


    def explain_instance(self, text: str, num_features: int = 10, 
                        num_samples: int = 100) -> Tuple[List[Tuple[str, float]], float]:
        """
        Generate LIME explanation for a single instance
        
        Args:
            text: Input text to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME
            
        Returns:
            Tuple of (feature_importance_list, prediction_probability)
        """
        # Get original prediction
        original_prob = self._predict_proba([text])[0][1]  # Spam probability
        
        # Generate LIME explanation
        explanation = self.lime_explainer.explain_instance(
            text,
            self._predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=[1]  # Explain spam class
        )
        
        # Extract feature importances
        feature_importance = explanation.as_list(label=1)
        
        return feature_importance, original_prob


    def _get_word_importance_ranking(self, feature_importance: List[Tuple[str, float]], 
                                   absolute: bool = True) -> List[Tuple[str, float]]:
        """
        Get word importance ranking from LIME explanation
        
        Args:
            feature_importance: List of (word, importance) tuples
            absolute: Whether to use absolute values for ranking
            
        Returns:
            List of (word, importance_score) tuples sorted by importance
        """
        if absolute:
            ranking = [(word, abs(score)) for word, score in feature_importance]
        else:
            ranking = feature_importance.copy()
        
        # Sort by importance (descending)
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking


    def _create_masked_text(self, text: str, words_to_remove: List[str], 
                          mask_strategy: str = 'remove') -> str:
        """
        Create masked version of text by removing/replacing specified words
        
        Args:
            text: Original text
            words_to_remove: List of words to mask
            mask_strategy: How to mask ('remove', 'mask', 'random')
            
        Returns:
            Masked text
        """
        if mask_strategy == 'remove':
            # Simply remove the words using case-insensitive matching
            masked_text = text
            for word in words_to_remove:
                # Use word boundaries to avoid partial matches, case-insensitive
                pattern = r'\b' + re.escape(word) + r'\b'
                masked_text = re.sub(pattern, '', masked_text, flags=re.IGNORECASE)
            
            # Clean up extra spaces
            masked_text = ' '.join(masked_text.split())
            
        elif mask_strategy == 'mask':
            # Replace with [MASK] token
            masked_text = text
            for word in words_to_remove:
                pattern = r'\b' + re.escape(word) + r'\b'
                masked_text = re.sub(pattern, '[MASK]', masked_text, flags=re.IGNORECASE)
                
        elif mask_strategy == 'random':
            # Replace with random words
            vocab_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
            masked_text = text
            for word in words_to_remove:
                random_word = random.choice(vocab_words)
                pattern = r'\b' + re.escape(word) + r'\b'
                masked_text = re.sub(pattern, random_word, masked_text, flags=re.IGNORECASE)
        
        # Ensure we return something meaningful
        if not masked_text.strip():
            return "empty text"
        
        return masked_text


    def compute_auc_deletion(self, text: str, steps: int = 20, 
                           mask_strategy: str = 'remove') -> float:
        """
        Compute AUC-Del: Area under deletion curve
        Lower values indicate better explanations
        
        Args:
            text: Original text
            steps: Number of deletion steps
            mask_strategy: How to mask words ('remove', 'mask', 'random')
            
        Returns:
            AUC-Del score (lower is better)
        """
        # Get LIME explanation
        feature_importance, original_pred = self.explain_instance(text)
        word_ranking = self._get_word_importance_ranking(feature_importance, absolute=True)
        
        if not word_ranking:
            return 0.0
        
        # Compute predictions for progressive deletion
        predictions = [original_pred]
        x_values = [0]
        
        words_to_remove = []
        step_size = max(1, len(word_ranking) // steps)
        
        for i in range(0, len(word_ranking), step_size):
            # Add next batch of most important words to removal list
            batch_end = min(i + step_size, len(word_ranking))
            words_to_remove.extend([word for word, _ in word_ranking[i:batch_end]])
            
            # Create masked text
            masked_text = self._create_masked_text(text, words_to_remove, mask_strategy)
            
            # Get prediction for masked text
            try:
                masked_pred = self._predict_proba([masked_text])[0][1]
            except:
                masked_pred = 0.0  # Fallback for invalid text
            
            predictions.append(masked_pred)
            x_values.append(len(words_to_remove))
        
        # Normalize x-values to [0, 1]
        max_words = len(word_ranking)
        if max_words > 0:
            x_normalized = [x / max_words for x in x_values]
        else:
            x_normalized = x_values
        
        # Calculate AUC (lower is better for deletion)
        if len(x_normalized) > 1 and len(predictions) > 1:
            auc_del = auc(x_normalized, predictions)
        else:
            auc_del = 0.0
            
        return auc_del


    def compute_auc_insertion(self, text: str, steps: int = 20) -> float:
        """
        Compute AUC-Ins: Area under insertion curve
        Higher values indicate better explanations
        
        Args:
            text: Original text
            steps: Number of insertion steps
            
        Returns:
            AUC-Ins score (higher is better)
        """
        # Get LIME explanation
        feature_importance, _ = self.explain_instance(text)
        word_ranking = self._get_word_importance_ranking(feature_importance, absolute=True)
        
        if not word_ranking:
            return 0.0
        
        # Start with completely masked text (all important words removed)
        all_important_words = [word for word, _ in word_ranking]
        baseline_text = self._create_masked_text(text, all_important_words, 'remove')
        
        try:
            baseline_pred = self._predict_proba([baseline_text])[0][1]
        except:
            baseline_pred = 0.0
        
        # Compute predictions for progressive insertion
        predictions = [baseline_pred]
        x_values = [0]
        
        words_to_keep = []
        step_size = max(1, len(word_ranking) // steps)
        
        for i in range(0, len(word_ranking), step_size):
            # Add next batch of most important words
            batch_end = min(i + step_size, len(word_ranking))
            words_to_keep.extend([word for word, _ in word_ranking[i:batch_end]])
            
            # Create text with only these words kept (others removed)
            words_to_remove = [word for word in all_important_words if word not in words_to_keep]
            partially_masked_text = self._create_masked_text(text, words_to_remove, 'remove')
            
            # Get prediction
            try:
                pred = self._predict_proba([partially_masked_text])[0][1]
            except:
                pred = baseline_pred
            
            predictions.append(pred)
            x_values.append(len(words_to_keep))
        
        # Normalize x-values to [0, 1]
        max_words = len(word_ranking)
        if max_words > 0:
            x_normalized = [x / max_words for x in x_values]
        else:
            x_normalized = x_values
        
        # Calculate AUC (higher is better for insertion)
        if len(x_normalized) > 1 and len(predictions) > 1:
            auc_ins = auc(x_normalized, predictions)
        else:
            auc_ins = 0.0
            
        return auc_ins


    def compute_comprehensiveness(self, text: str, k: int = 5) -> float:
        """
        Compute comprehensiveness: prediction change when removing top-k features
        Higher values indicate better explanations
        
        Args:
            text: Original text
            k: Number of top features to remove
            
        Returns:
            Comprehensiveness score (higher is better)
        """
        # Get LIME explanation
        feature_importance, original_pred = self.explain_instance(text)
        word_ranking = self._get_word_importance_ranking(feature_importance, absolute=True)
        
        # Get top-k most important words
        top_k_words = [word for word, _ in word_ranking[:k]]
        
        if not top_k_words:
            return 0.0
        
        # Create text with top-k words removed
        masked_text = self._create_masked_text(text, top_k_words, 'remove')
        
        # Get prediction without top-k features
        try:
            masked_pred = self._predict_proba([masked_text])[0][1]
        except:
            masked_pred = 0.0
        
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
            perturbation_prob: Probability of perturbing each word
            
        Returns:
            Average Jaccard similarity (higher is better)
        """
        # Get original explanation
        feature_importance, _ = self.explain_instance(text)
        original_ranking = self._get_word_importance_ranking(feature_importance, absolute=True)
        original_top_k = set([word for word, _ in original_ranking[:k]])
        
        jaccard_scores = []
        
        # Tokenize text for perturbation
        words = text.split()
        
        for _ in range(num_perturbations):
            # Create perturbed version by randomly replacing some words
            perturbed_words = words.copy()
            
            # Simple vocabulary for replacement
            replacement_vocab = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with',
                               'for', 'on', 'as', 'you', 'are', 'this', 'have', 'can', 'will', 'be']
            
            for i in range(len(perturbed_words)):
                if random.random() < perturbation_prob:
                    # Replace with a random word
                    perturbed_words[i] = random.choice(replacement_vocab)
            
            # Reconstruct text
            perturbed_text = ' '.join(perturbed_words)
            
            try:
                # Get explanation for perturbed text
                perturbed_feature_importance, _ = self.explain_instance(perturbed_text)
                perturbed_ranking = self._get_word_importance_ranking(perturbed_feature_importance, absolute=True)
                perturbed_top_k = set([word for word, _ in perturbed_ranking[:k]])
                
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


    def evaluate_explanation_quality(self, text: str, verbose: bool = True) -> Dict[str, float]:
        """
        Compute all explanation quality metrics for a given text
        
        Args:
            text: Input text
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary containing all metric scores
        """
        if verbose:
            print(f"Evaluating LIME explanation quality for text: '{text[:50]}...'" if len(text) > 50 else f"Evaluating LIME explanation quality for text: '{text}'")
        
        metrics = {}
        
        try:
            # AUC-Del (lower is better)
            if verbose:
                print("Computing AUC-Del...")
            overall_start_time = pd.Timestamp.now()
            start_time = pd.Timestamp.now()
            metrics['auc_deletion'] = self.compute_auc_deletion(text)
            metrics['auc_deletion_time'] = (pd.Timestamp.now() - start_time).total_seconds()
            
            # AUC-Ins (higher is better)
            if verbose:
                print("Computing AUC-Ins...")
            start_time = pd.Timestamp.now()
            metrics['auc_insertion'] = self.compute_auc_insertion(text)
            metrics['auc_insertion_time'] = (pd.Timestamp.now() - start_time).total_seconds()
            
            # Comprehensiveness (higher is better)
            if verbose:
                print("Computing Comprehensiveness...")
            start_time = pd.Timestamp.now()
            metrics['comprehensiveness'] = self.compute_comprehensiveness(text)
            metrics['comprehensiveness_time'] = (pd.Timestamp.now() - start_time).total_seconds()
            
            # Jaccard Stability (higher is better)
            if verbose:
                print("Computing Jaccard Stability...")
            start_time = pd.Timestamp.now()
            metrics['jaccard_stability'] = self.compute_jaccard_stability(text)
            metrics['jaccard_stability_time'] = (pd.Timestamp.now() - start_time).total_seconds()
            metrics['total_time'] = (pd.Timestamp.now() - overall_start_time).total_seconds()
            
            if verbose:
                print("\n" + "=" * 50)
                print("LIME EXPLANATION QUALITY METRICS")
                print("=" * 50)
                print(f"AUC-Deletion:     {metrics['auc_deletion']:.4f} (lower is better)")
                print(f"AUC-Insertion:    {metrics['auc_insertion']:.4f} (higher is better)")
                print(f"Comprehensiveness: {metrics['comprehensiveness']:.4f} (higher is better)")
                print(f"Jaccard Stability: {metrics['jaccard_stability']:.4f} (higher is better)")
                print("\nTiming Information:")
                print(f"AUC-Deletion Time: {metrics['auc_deletion_time']:.2f} seconds")
                print(f"AUC-Insertion Time: {metrics['auc_insertion_time']:.2f} seconds")
                print(f"Comprehensiveness Time: {metrics['comprehensiveness_time']:.2f} seconds")
                print(f"Jaccard Stability Time: {metrics['jaccard_stability_time']:.2f} seconds")
                print(f"Total Time: {metrics['total_time']:.2f} seconds")
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


    def plot_deletion_insertion_curves(self, text: str, steps: int = 20, save_path: str = None):
        """
        Plot deletion and insertion curves for visualization
        
        Args:
            text: Input text
            steps: Number of steps for the curves
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        # Get LIME explanation
        feature_importance, original_pred = self.explain_instance(text)
        word_ranking = self._get_word_importance_ranking(feature_importance, absolute=True)
        
        # Deletion curve
        del_predictions = [original_pred]
        del_x = [0]
        words_to_remove = []
        step_size = max(1, len(word_ranking) // steps)
        
        for i in range(0, len(word_ranking), step_size):
            batch_end = min(i + step_size, len(word_ranking))
            words_to_remove.extend([word for word, _ in word_ranking[i:batch_end]])
            
            masked_text = self._create_masked_text(text, words_to_remove, 'remove')
            
            try:
                masked_pred = self._predict_proba([masked_text])[0][1]
            except:
                masked_pred = 0.0
            
            del_predictions.append(masked_pred)
            del_x.append(len(words_to_remove))
        
        # Insertion curve
        all_important_words = [word for word, _ in word_ranking]
        baseline_text = self._create_masked_text(text, all_important_words, 'remove')
        
        try:
            baseline_pred = self._predict_proba([baseline_text])[0][1]
        except:
            baseline_pred = 0.0
        
        ins_predictions = [baseline_pred]
        ins_x = [0]
        words_to_keep = []
        
        for i in range(0, len(word_ranking), step_size):
            batch_end = min(i + step_size, len(word_ranking))
            words_to_keep.extend([word for word, _ in word_ranking[i:batch_end]])
            
            words_to_remove = [word for word in all_important_words if word not in words_to_keep]
            partially_masked_text = self._create_masked_text(text, words_to_remove, 'remove')
            
            try:
                pred = self._predict_proba([partially_masked_text])[0][1]
            except:
                pred = baseline_pred
            
            ins_predictions.append(pred)
            ins_x.append(len(words_to_keep))
        
        # Normalize x-values
        max_del_words = len(word_ranking) if len(word_ranking) > 0 else 1
        max_ins_words = len(word_ranking) if len(word_ranking) > 0 else 1
        
        del_x_norm = [x / max_del_words for x in del_x]
        ins_x_norm = [x / max_ins_words for x in ins_x]
        
        # Create plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(del_x_norm, del_predictions, 'r-', linewidth=2, label='Deletion Curve')
        plt.xlabel('Fraction of Features Removed')
        plt.ylabel('Prediction Score')
        plt.title('LIME Deletion Curve\n(Lower AUC = Better)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(ins_x_norm, ins_predictions, 'g-', linewidth=2, label='Insertion Curve')
        plt.xlabel('Fraction of Features Added')
        plt.ylabel('Prediction Score')
        plt.title('LIME Insertion Curve\n(Higher AUC = Better)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def demonstrate_bert_lime_quality_metrics(model, tokenizer, test_text: str, device: str = 'cpu'):
    """
    Demonstrate how to use the BERT LIME explanation quality metrics
    
    Args:
        model: SpamBERT model instance
        tokenizer: BERT tokenizer
        test_text: Text to analyze
        device: Device to run on
    """
    print("Initializing BERT LIME quality metrics calculator...")
    lime_evaluator = BertLimeExplainer(model, tokenizer, device)
    
    print("Computing explanation quality metrics using LIME...")
    metrics = lime_evaluator.evaluate_explanation_quality(test_text)
    
    print("Plotting deletion and insertion curves...")
    lime_evaluator.plot_deletion_insertion_curves(test_text)
    
    return metrics
