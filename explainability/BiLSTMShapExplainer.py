from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import shap
import spacy
import warnings
from utils.functions import encode


class BiLSTMShapExplainer:
    def __init__(self, model, word_to_idx: Dict[str, int], idx_to_word: Dict[int, str], max_length: int = 256, device: str = 'cpu'):
        """
        Initialize SHAP Kernel explainer for BiLSTM model

        Args:
            model: Trained BiLSTM model
            word_to_idx: Dictionary mapping words to indices
            idx_to_word: Dictionary mapping indices to words
            max_length: Maximum sequence length for padding/truncation
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.model.eval()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.max_length = max_length
        self.device = device

        # Load spaCy tokenizer
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            print("spaCy model 'en_core_web_sm' not found. Please install it with:")
            print("python -m spacy download en_core_web_sm")
            raise

        # Initialize SHAP Kernel explainer (will be set up later with background data)
        self.explainer = None
        self.background_data = None
        self.background_vectors = None
        self.max_features = None

    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text using spaCy
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        doc = self.nlp(text.lower())
        return [token.text for token in doc]

    def text_to_sequence(self, text: str) -> torch.Tensor:
        """
        Convert text to sequence of token indices
        
        Args:
            text: Input text
            
        Returns:
            Tensor of token indices
        """
        return torch.tensor(encode(text, self.word_to_idx, self.max_length))

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
                
                # Convert text to sequence
                sequence = self.text_to_sequence(text).unsqueeze(0).to(self.device)
                
                # Get prediction (only the probability, not attention weights)
                output = self.model(sequence)
                if isinstance(output, tuple):
                    pred = output[0].item()
                else:
                    pred = output.item()
                
                predictions.append(pred)
        
        return np.array(predictions)

    def setup_explainer(self, background_texts: List[str], nsamples: int = 100):
        """
        Setup SHAP Kernel explainer with background data
        
        Args:
            background_texts: List of background texts for SHAP
            nsamples: Number of samples for SHAP explanation
        """
        print(f"Setting up SHAP Kernel explainer with {len(background_texts)} background samples...")
        
        # Sample background data if too large
        if len(background_texts) > 50:  # Kernel explainer works better with smaller background
            background_sample = np.random.choice(
                background_texts, size=50, replace=False
            ).tolist()
        else:
            background_sample = background_texts
        
        # Store background data
        self.background_data = background_sample
        
        # Convert background texts to token indices for the model
        background_vectors = []
        for text in background_sample:
            tokens = self.tokenize_text(text)
            # Create a feature vector where each position represents presence of tokens
            # For simplicity, we'll use the first few tokens as features
            max_features = min(50, max(len(self.tokenize_text(t)) for t in background_sample))
            feature_vector = np.zeros(max_features)
            for i, token in enumerate(tokens[:max_features]):
                feature_vector[i] = 1  # Binary feature: token present
            background_vectors.append(feature_vector)
        
        self.background_vectors = np.array(background_vectors)
        self.max_features = max_features
        
        # Create prediction function that works with feature vectors
        def model_wrapper(feature_vectors):
            """Convert feature vectors back to texts and get predictions"""
            predictions = []
            
            for feature_vector in feature_vectors:
                # Reconstruct text from feature vector using helper method
                if hasattr(feature_vector, '__len__') and len(feature_vector) > 0:
                    text = self.feature_vector_to_text(feature_vector)
                else:
                    text = ""
                
                # Get prediction
                if text.strip():
                    pred = self.prediction_function([text])[0]
                else:
                    pred = 0.0  # Default prediction for empty text
                
                predictions.append(pred)
            
            return np.array(predictions)
        
        # Create SHAP Kernel explainer
        self.explainer = shap.KernelExplainer(
            model=model_wrapper,
            data=self.background_vectors,
            feature_names=[f"token_{i}" for i in range(self.max_features)]
        )
        
        print("SHAP Kernel explainer setup complete!")

    def explain_prediction(self, text: str, nsamples: int = 100) -> np.ndarray:
        """
        Generate SHAP explanation for a text using Kernel explainer
        
        Args:
            text: Text to explain
            nsamples: Number of samples for explanation
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized. Call setup_explainer() first.")
        
        # Tokenize the input text
        tokens = self.tokenize_text(text)
        
        # Convert text to feature vector using helper method
        feature_vector = self.text_to_feature_vector(text)
        
        # Reshape for SHAP (expects 2D array)
        feature_vector = feature_vector.reshape(1, -1)
        
        # Suppress warnings for cleaner output
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # Get SHAP values using Kernel explainer
            shap_values = self.explainer.shap_values(
                feature_vector, 
                nsamples=nsamples,
                silent=True
            )
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification might return list
                shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
            
            # Ensure we have the right shape
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Trim to actual number of tokens in the text
            num_tokens = len(tokens)
            result = np.zeros((1, num_tokens))
            
            # Copy SHAP values for actual tokens
            for i in range(min(num_tokens, shap_values.shape[1])):
                result[0, i] = shap_values[0, i]
        
        return result

    def get_token_importance_ranking(self, text: str, shap_values: np.ndarray, 
                                   absolute: bool = True) -> List[Tuple[int, float, str]]:
        """
        Get token importance ranking from SHAP values
        
        Args:
            text: Original text
            shap_values: SHAP values array
            absolute: Whether to use absolute values for ranking
            
        Returns:
            List of (token_index, importance_score, token) tuples sorted by importance
        """
        tokens = self.tokenize_text(text)
        
        # Get importance scores for valid tokens only
        token_importance = []
        for i, token in enumerate(tokens):
            if i < len(shap_values[0]):
                importance = abs(shap_values[0][i]) if absolute else shap_values[0][i]
                token_importance.append((i, importance, token))
        
        # Sort by importance (descending)
        token_importance.sort(key=lambda x: x[1], reverse=True)
        return token_importance

    def visualize_explanation(self, text: str, shap_values: np.ndarray, 
                            save_path: str = None, max_tokens: int = 20):
        """
        Visualize SHAP explanation
        
        Args:
            text: Original text
            shap_values: SHAP values
            save_path: Path to save visualization
            max_tokens: Maximum number of tokens to display
        """
        try:
            import matplotlib.pyplot as plt
            
            tokens = self.tokenize_text(text)[:max_tokens]
            values = shap_values[0][:len(tokens)]
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            colors = ['red' if v < 0 else 'green' for v in values]
            bars = plt.bar(range(len(tokens)), np.abs(values), color=colors, alpha=0.7)
            
            plt.xlabel('Tokens')
            plt.ylabel('|SHAP Value|')
            plt.title('BiLSTM SHAP Token Importance')
            plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")
        except Exception as e:
            print(f"Error creating visualization: {e}")

    def text_to_feature_vector(self, text: str) -> np.ndarray:
        """
        Convert text to feature vector for SHAP Kernel explainer
        
        Args:
            text: Input text
            
        Returns:
            Feature vector representing token presence
        """
        tokens = self.tokenize_text(text)
        feature_vector = np.zeros(self.max_features)
        
        for i, token in enumerate(tokens[:self.max_features]):
            feature_vector[i] = 1  # Binary feature: token present
            
        return feature_vector
    
    def feature_vector_to_text(self, feature_vector: np.ndarray, reference_text: str = None) -> str:
        """
        Convert feature vector back to text using masking approach
        
        Args:
            feature_vector: Binary feature vector
            reference_text: Reference text to use for reconstruction
            
        Returns:
            Reconstructed text
        """
        if reference_text is None:
            # Use first background text as reference
            reference_text = self.background_data[0] if self.background_data else ""
        
        reference_tokens = self.tokenize_text(reference_text)
        masked_tokens = []
        
        for i in range(min(len(reference_tokens), len(feature_vector))):
            if feature_vector[i] > 0.5:  # Token is "present"
                masked_tokens.append(reference_tokens[i])
        
        return " ".join(masked_tokens)
