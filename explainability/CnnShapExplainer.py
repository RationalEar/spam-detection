from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shap
import spacy
import torch


class CnnShapExplainer:
    def __init__(self, model, word_to_idx: Dict[str, int], idx_to_word: Dict[int, str],
                 max_length: int = 256, device: str = 'cpu'):
        """
        Initialize SHAP explainer for SpamCNN model

        Args:
            model: Trained SpamCNN model
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

        # Initialize SHAP explainer (will be set up later with background data)
        self.explainer = None

    def preprocess_text(self, text: str) -> List[str]:
        """
        Tokenize and preprocess text using spaCy

        Args:
            text: Input text string

        Returns:
            List of preprocessed tokens
        """
        doc = self.nlp(text.lower())
        tokens = []
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                tokens.append(token.lemma_)
        return tokens

    def text_to_sequence(self, text: str) -> torch.Tensor:
        """
        Convert text to sequence of token indices

        Args:
            text: Input text string

        Returns:
            Tensor of token indices
        """
        tokens = self.preprocess_text(text)
        sequence = []

        for token in tokens:
            if token in self.word_to_idx:
                sequence.append(self.word_to_idx[token])
            else:
                # Use unknown token if available, otherwise skip
                if '<UNK>' in self.word_to_idx:
                    sequence.append(self.word_to_idx['<UNK>'])

        # Pad or truncate to max_length
        if len(sequence) < self.max_length:
            sequence.extend([0] * (self.max_length - len(sequence)))  # Pad with 0
        else:
            sequence = sequence[:self.max_length]  # Truncate

        return torch.tensor(sequence, dtype=torch.long)

    def _model_predict_from_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Prediction function for SHAP explainer that takes sequence arrays

        Args:
            sequences: Numpy array of token sequences (batch_size, seq_len)

        Returns:
            Array of prediction probabilities
        """
        # Convert numpy array to torch tensor
        batch = torch.tensor(sequences, dtype=torch.long).to(self.device)

        # Get predictions
        with torch.no_grad():
            predictions = self.model.predict(batch)

        return predictions.cpu().numpy()

    def model_predict(self, texts: List[str]) -> np.ndarray:
        """
        Prediction function for SHAP explainer

        Args:
            texts: List of text strings

        Returns:
            Array of prediction probabilities
        """
        sequences = []
        for text in texts:
            seq = self.text_to_sequence(text)
            sequences.append(seq)

        # Stack sequences into batch
        batch = torch.stack(sequences).to(self.device)

        # Get predictions
        with torch.no_grad():
            predictions = self.model.predict(batch)

        return predictions.cpu().numpy()

    def setup_explainer(self, background_texts: List[str], n_background: int = 100):
        """
        Set up SHAP KernelExplainer with background samples

        Args:
            background_texts: List of background text samples
            n_background: Number of background samples to use
        """
        # Sample background texts if we have more than needed
        if len(background_texts) > n_background:
            background_sample = np.random.choice(
                background_texts, size=n_background, replace=False
            ).tolist()
        else:
            background_sample = background_texts

        print(f"Setting up SHAP explainer with {len(background_sample)} background samples...")

        # Convert background texts to sequences (the format our model expects)
        background_sequences = []
        for text in background_sample:
            seq = self.text_to_sequence(text)
            background_sequences.append(seq.numpy())

        # Convert to numpy array for SHAP
        background_data = np.array(background_sequences)

        print(f"Background data shape: {background_data.shape}")

        # Initialize SHAP KernelExplainer with a wrapper function
        self.explainer = shap.KernelExplainer(
            self._model_predict_from_sequences,
            background_data
        )

        print("SHAP explainer ready!")

    def explain_prediction(self, text: str, n_samples: int = 100) -> shap.Explanation:
        """
        Generate SHAP explanation for a single text

        Args:
            text: Input text to explain
            n_samples: Number of samples for KernelSHAP

        Returns:
            SHAP explanation object
        """
        if self.explainer is None:
            raise ValueError("Explainer not set up. Call setup_explainer() first.")

        # Convert text to sequence for explanation
        sequence = self.text_to_sequence(text).numpy()
        sequence_array = np.array([sequence])  # Add batch dimension

        # Generate explanation
        shap_values = self.explainer.shap_values(sequence_array, nsamples=n_samples)

        return shap_values

    def plot_explanation(self, text: str, shap_values: np.ndarray,
                         max_display: int = 20, save_path: str = None,
                         subject: str = "Spam Detection", label: str = None):
        """
        Plot SHAP explanation as a waterfall plot

        Args:
            text: Original text
            shap_values: SHAP values from explain_prediction
            max_display: Maximum number of features to display
            save_path: Path to save the plot (optional)
            subject: Subject of the email (for title)
            label: Label for the explanation (e.g., "spam" or "ham")
        """
        # Tokenize the text to get individual words for display
        tokens = self.preprocess_text(text)

        # Create explanation object for plotting
        if len(tokens) > max_display:
            # Show top contributing tokens
            token_importance = [(abs(val), i, token) for i, (val, token) in
                                enumerate(zip(shap_values[0], tokens))]
            token_importance.sort(reverse=True)

            selected_indices = [item[1] for item in token_importance[:max_display]]
            selected_tokens = [tokens[i] for i in selected_indices]
            selected_shap_values = [shap_values[0][i] for i in selected_indices]
        else:
            selected_tokens = tokens
            selected_shap_values = shap_values[0][:len(tokens)]

        # Create the plot
        plt.figure(figsize=(10, max(6, int(len(selected_tokens) * 0.3))))

        # Sort by SHAP value for better visualization
        sorted_data = sorted(zip(selected_shap_values, selected_tokens),
                             key=lambda x: x[0])
        sorted_values, sorted_tokens = zip(*sorted_data)

        colors = ['red' if val > 0 else 'green' for val in sorted_values]

        plt.barh(range(len(sorted_tokens)), sorted_values, color=colors, alpha=0.7)
        plt.yticks(range(len(sorted_tokens)), sorted_tokens)
        plt.xlabel('SHAP Value (Impact on Spam Probability)')
        plt.title(f"SHAP Explanation for {label}: {subject}")
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        # Add legend
        plt.legend(['Decreases Spam Probability', 'Increases Spam Probability'],
                   loc='best')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def get_top_features(self, text: str, shap_values: np.ndarray,
                         n_features: int = 10) -> List[Tuple[str, float]]:
        """
        Get top contributing features from SHAP explanation

        Args:
            text: Original text
            shap_values: SHAP values from explain_prediction
            n_features: Number of top features to return

        Returns:
            List of (token, shap_value) tuples sorted by absolute importance
        """
        tokens = self.preprocess_text(text)

        # Pair tokens with their SHAP values
        token_shap_pairs = [(token, shap_val) for token, shap_val in
                            zip(tokens, shap_values[0][:len(tokens)])]

        # Sort by absolute SHAP value
        token_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        return token_shap_pairs[:n_features]
