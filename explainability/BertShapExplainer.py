from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import shap
import warnings
from transformers import BertTokenizer


class BertShapExplainer:
    def __init__(self, model, tokenizer: BertTokenizer, max_length: int = 512, device: str = 'cpu'):
        """
        Initialize SHAP Kernel explainer for BERT model

        Args:
            model: Trained BERT model (SpamBERT)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length for padding/truncation
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        # Initialize SHAP Kernel explainer (will be set up later with background data)
        self.explainer = None
        self.background_data = None
        self.background_vectors = None
        self.max_features = None

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text using BERT tokenizer
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def text_to_sequence(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Convert text to BERT input format
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing input_ids, attention_mask, and token_type_ids
        """
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device),
            'token_type_ids': encoded.get('token_type_ids', None)
        }

    def prediction_function(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Prediction function for SHAP explainer
        
        Args:
            texts: List or array of text strings
            
        Returns:
            Array of prediction probabilities
        """
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                if isinstance(text, (list, np.ndarray)):
                    # If input is already tokenized, join it back to text
                    text = ' '.join(str(t) for t in text)
                
                # Convert text to BERT input format
                inputs = self.text_to_sequence(text)
                
                # Get prediction (only the probability, not attention weights)
                output = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs['token_type_ids']
                )
                
                if isinstance(output, tuple):
                    pred = output[0].item()
                else:
                    pred = output.item()
                
                predictions.append(pred)
        
        return np.array(predictions)

    def setup_explainer(self, background_texts: List[str], max_features: int = None):
        """
        Set up SHAP explainer with background data
        
        Args:
            background_texts: List of background texts for SHAP
            max_features: Maximum number of features to consider
        """
        print(f"Setting up SHAP explainer with {len(background_texts)} background samples...")
        
        # Use a subset of background data if it's too large
        if len(background_texts) > 100:
            background_texts = background_texts[:100]
            print(f"Using subset of {len(background_texts)} background samples for efficiency")
        
        self.background_data = background_texts
        self.max_features = max_features
        
        # For text data, we need to use a different approach
        # We'll create the explainer when needed in explain_text methods
        self.explainer = "ready"  # Flag to indicate setup is complete
        
        print("SHAP explainer setup complete")

    def explain_text(self, text: str, max_evals: int = 1000) -> Tuple[List[str], np.ndarray]:
        """
        Generate SHAP explanation for a text
        
        Args:
            text: Text to explain
            max_evals: Maximum number of evaluations for SHAP
            
        Returns:
            Tuple of (tokens, shap_values)
        """
        if self.explainer != "ready":
            raise ValueError("SHAP explainer not set up. Call setup_explainer() first.")
        
        # Use word-level explanation for simplicity
        words = text.split()
        
        # Limit words if max_features is set
        if self.max_features and len(words) > self.max_features:
            words = words[:self.max_features]
            text = ' '.join(words)
        
        print(f"Explaining text with {len(words)} words...")
        
        # Create a word-based explanation function
        def word_prediction_function(word_masks):
            """Prediction function that operates on word-level masks"""
            predictions = []
            for mask in word_masks:
                # Create masked text based on word mask
                if len(mask) != len(words):
                    # Handle size mismatch by truncating or padding
                    if len(mask) > len(words):
                        mask = mask[:len(words)]
                    else:
                        mask = list(mask) + [0] * (len(words) - len(mask))
                
                masked_words = [word if mask[i] > 0.5 else '[MASK]' for i, word in enumerate(words)]
                masked_text = ' '.join(masked_words)
                pred = self.prediction_function([masked_text])[0]
                predictions.append(pred)
            return np.array(predictions)
        
        # Create baseline (all words present)
        baseline = np.ones((1, len(words)))
        
        # Create a simple explainer for this specific text
        from shap import Explainer
        explainer = shap.Explainer(word_prediction_function, baseline)
        
        # Get SHAP values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer(baseline, max_evals=max_evals)
        
        return words, shap_values.values

    def explain_text_simple(self, text: str, nsamples: int = 500) -> Tuple[List[str], np.ndarray]:
        """
        Generate SHAP explanation for a text with simplified approach
        
        Args:
            text: Text to explain
            nsamples: Number of samples for SHAP estimation
            
        Returns:
            Tuple of (tokens, shap_values)
        """
        if self.explainer != "ready":
            raise ValueError("SHAP explainer not set up. Call setup_explainer() first.")
        
        # Use word-level explanation for simplicity
        words = text.split()
        
        # Limit words if max_features is set  
        if self.max_features and len(words) > self.max_features:
            words = words[:self.max_features]
            text = ' '.join(words)
        
        if len(words) == 0:
            return [], np.array([])
        
        print(f"Explaining text with {len(words)} words using {nsamples} samples...")
        
        # Create a simplified word-based explanation function
        def simple_word_prediction_function(word_indices):
            """Prediction function that operates on word indices"""
            predictions = []
            for indices in word_indices:
                if len(indices) == 0:
                    # Empty text
                    masked_text = ""
                else:
                    # Select only the specified words
                    selected_words = [words[int(i)] for i in indices if int(i) < len(words)]
                    masked_text = ' '.join(selected_words)
                
                if masked_text.strip():
                    pred = self.prediction_function([masked_text])[0]
                else:
                    pred = 0.5  # Neutral prediction for empty text
                predictions.append(pred)
            return np.array(predictions)
        
        # Use a different approach: permutation-based SHAP
        try:
            import itertools
            import random
            
            # Get baseline prediction (full text)
            baseline_pred = self.prediction_function([text])[0]
            
            # Compute SHAP values using sampling approach
            shap_values = np.zeros(len(words))
            
            # Sample different combinations of words
            for _ in range(min(nsamples, 2**len(words))):
                # Randomly select a subset of words
                num_words_to_select = random.randint(0, len(words))
                selected_indices = random.sample(range(len(words)), num_words_to_select)
                
                # Create text with selected words
                if selected_indices:
                    selected_words = [words[i] for i in sorted(selected_indices)]
                    partial_text = ' '.join(selected_words)
                    partial_pred = self.prediction_function([partial_text])[0]
                else:
                    partial_pred = 0.5
                
                # Update SHAP values (simplified Shapley value approximation)
                contribution = (baseline_pred - partial_pred) / len(words)
                for i in selected_indices:
                    shap_values[i] += contribution / nsamples
            
            return words, shap_values.reshape(1, -1)
            
        except Exception as e:
            print(f"Warning: Simplified SHAP computation failed: {e}")
            # Fallback: use feature importance based on word removal
            shap_values = []
            
            for i, word in enumerate(words):
                # Remove this word and see impact on prediction
                remaining_words = words[:i] + words[i+1:]
                remaining_text = ' '.join(remaining_words)
                
                if remaining_text.strip():
                    remaining_pred = self.prediction_function([remaining_text])[0]
                else:
                    remaining_pred = 0.5
                
                # SHAP value is the difference
                shap_value = baseline_pred - remaining_pred
                shap_values.append(shap_value)
            
            return words, np.array(shap_values).reshape(1, -1)

    def get_top_features(self, tokens: List[str], shap_values: np.ndarray, 
                        top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top-k most important features from SHAP values
        
        Args:
            tokens: List of tokens
            shap_values: SHAP values array
            top_k: Number of top features to return
            
        Returns:
            List of (token, importance) tuples
        """
        if len(shap_values.shape) > 1:
            # Take the first instance if batch
            values = shap_values[0]
        else:
            values = shap_values
        
        # Create token-importance pairs
        token_importance = []
        for i, token in enumerate(tokens):
            if i < len(values):
                importance = abs(values[i])
                token_importance.append((token, importance))
        
        # Sort by importance and return top-k
        token_importance.sort(key=lambda x: x[1], reverse=True)
        return token_importance[:top_k]

    def visualize_explanation(self, tokens: List[str], shap_values: np.ndarray, 
                            save_path: str = None):
        """
        Visualize SHAP explanation
        
        Args:
            tokens: List of tokens
            shap_values: SHAP values array
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if len(shap_values.shape) > 1:
                values = shap_values[0]
            else:
                values = shap_values
            
            # Limit to reasonable number of tokens for visualization
            if len(tokens) > 20:
                top_indices = np.argsort(np.abs(values))[-20:]
                tokens = [tokens[i] for i in top_indices]
                values = values[top_indices]
            
            # Create horizontal bar plot
            plt.figure(figsize=(10, max(6, len(tokens) * 0.3)))
            colors = ['red' if val < 0 else 'blue' for val in values]
            bars = plt.barh(range(len(tokens)), values, color=colors, alpha=0.7)
            
            plt.yticks(range(len(tokens)), tokens)
            plt.xlabel('SHAP Value')
            plt.title('SHAP Feature Importance')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                plt.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"SHAP visualization saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")


def demonstrate_bert_shap_explanation(model, tokenizer, test_texts: List[str], 
                                    background_texts: List[str], device: str = 'cpu'):
    """
    Demonstrate how to use the BERT SHAP explainer
    
    Args:
        model: SpamBERT model instance
        tokenizer: BERT tokenizer
        test_texts: Texts to explain
        background_texts: Background texts for SHAP
        device: Device to run on
    """
    print("Initializing BERT SHAP explainer...")
    explainer = BertShapExplainer(model, tokenizer, device=device)
    
    print("Setting up SHAP explainer with background data...")
    explainer.setup_explainer(background_texts)
    
    results = []
    for i, text in enumerate(test_texts):
        print(f"\nExplaining text {i+1}/{len(test_texts)}...")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Get explanation
        tokens, shap_values = explainer.explain_text_simple(text)
        
        # Get top features
        top_features = explainer.get_top_features(tokens, shap_values)
        
        print("Top contributing features:")
        for j, (token, importance) in enumerate(top_features[:5]):
            print(f"  {j+1}. '{token}': {importance:.4f}")
        
        # Visualize if requested
        explainer.visualize_explanation(tokens, shap_values)
        
        results.append({
            'text': text,
            'tokens': tokens,
            'shap_values': shap_values,
            'top_features': top_features
        })
    
    return results
