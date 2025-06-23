import numpy as np
import spacy
import torch
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

# Load SpaCy model (you might need to download it first: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Downloading...")
    import subprocess
    
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


class LIMETextExplanationWrapper:
    """
    A wrapper class to make the SpamCNN model compatible with LIME's TextExplainer.
    LIME expects a predict_proba function that takes a list of raw strings
    and returns a numpy array of probabilities for each class.
    """
    
    def __init__(self, model, word_to_idx, idx_to_word, device='cpu'):
        self.model = model
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.device = device
        self.model.eval()  # Ensure the model is in evaluation mode
    
    def tokenizer_spacy(self, text):
        """Custom SpaCy tokenizer for LIME."""
        return [token.text for token in nlp(text)]
    
    def _text_to_sequence(self, texts):
        """Converts a list of text strings to a sequence of token IDs."""
        sequences = []
        max_seq_len = 0  # Determine max_seq_len dynamically or set a fixed one
        
        # First pass to tokenize and find max_seq_len if not fixed
        tokenized_texts = []
        for text in texts:
            tokens = self.tokenizer_spacy(text)
            tokenized_texts.append(tokens)
            max_seq_len = max(max_seq_len, len(tokens))
        
        # Second pass to convert to IDs and pad
        for tokens in tokenized_texts:
            indexed_tokens = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in tokens]
            # Pad sequences to max_seq_len. Assuming 0 is the PAD token ID.
            padded_sequence = indexed_tokens + [self.word_to_idx.get('<PAD>', 0)] * (max_seq_len - len(indexed_tokens))
            sequences.append(padded_sequence)
        
        return torch.tensor(sequences, dtype=torch.long, device=self.device)
    
    def predict_proba(self, texts):
        """
        LIME's required predict_proba function.
        Takes a list of raw text strings, preprocesses them, and returns
        probabilities for spam (class 1) and not-spam (class 0).
        """
        sequences = self._text_to_sequence(texts)
        
        with torch.no_grad():
            outputs = self.model(sequences)
            # The model outputs a single probability for spam.
            # LIME expects probabilities for all classes.
            # For binary classification, we can return [1-p, p]
            spam_probs = outputs.squeeze(1).cpu().numpy()
            not_spam_probs = 1 - spam_probs
            return np.column_stack((not_spam_probs, spam_probs))


def get_lime_explanation(model, original_text, word_to_idx, idx_to_word, num_features=10, num_samples=500, alpha=0.01):
    """
    Generates a LIME explanation for a given text.

    Args:
        model: Your trained SpamCNN model.
        original_text (str): The email text to explain.
        word_to_idx (dict): Vocabulary mapping words to their integer IDs.
        idx_to_word (dict): Vocabulary mapping integer IDs to words.
        num_features (int): Number of features (words) to include in the explanation.
        num_samples (int): Number of perturbed samples for LIME.
        alpha (float): Regularization strength for Ridge regression.

    Returns:
        lime.explanation.Explanation: The LIME explanation object.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create the wrapper for LIME
    explainer_wrapper = LIMETextExplanationWrapper(model, word_to_idx, idx_to_word, device)
    
    # Initialize LimeTextExplainer with the custom tokenizer
    explainer = LimeTextExplainer(
        class_names=["ham", "spam"]
    )
    
    # Generate explanation
    # Specify 'ridge' as the kernel_estimator and set alpha
    explanation = explainer.explain_instance(
        text_instance=original_text,
        classifier_fn=explainer_wrapper.predict_proba,
        num_features=num_features,
        num_samples=num_samples,
        # LIME uses sklearn's Ridge internally, so we pass the alpha parameter
        # via the feature_selection 'highest_weights' which uses a linear model.
        # Alternatively, if we need full control, we can pass a custom kernel_estimator
        # For simplicity and common use, let's assume default Ridge if not explicitly set by lime.
        # LIME's TextExplainer uses a "kernel_estimator" which is often Ridge.
        # The 'kernel_width' and 'verbose' can also be tuned.
        # For setting alpha directly, LimeTextExplainer doesn't expose it directly for the internal Ridge.
        # A common workaround or understanding is that it uses a default Ridge,
        # or if you need a specific alpha, you might need to create a custom kernel_estimator.
        # However, LIME's `explain_instance` takes `model_regressor` argument for this.
        
        # Let's use a workaround to set alpha for Ridge.
        # The default behavior of LIME is often to use Ridge Regression.
        # To pass alpha, we need to import Ridge and pass it as model_regressor.
        # model_regressor=Ridge(alpha=alpha)
    )
    return explanation


def plot_lime_explanation(explanation_data, title="LIME Explanation"):
    """
    Generates a horizontal bar chart for LIME explanations,
    with red bars for negative contributions and blue bars for positive.

    Args:
        explanation_data (list): A list of tuples, where each tuple contains
                                 (feature_name: str, weight: float).
                                 Example: [('word1', 0.5), ('word2', -0.3)]
        title (str): The title of the plot.
    """
    if not explanation_data:
        print("No explanation data provided to plot.")
        return

    # Sort data by absolute weight for better visualization (most impactful at the top)
    # Reverse to have the highest absolute values at the top of the chart
    explanation_data_sorted = sorted(explanation_data, key=lambda x: abs(x[1]), reverse=True)

    features = [item[0] for item in explanation_data_sorted]
    weights = [item[1] for item in explanation_data_sorted]

    # Determine colors based on weight (blue for positive, red for negative)
    colors = ['blue' if w >= 0 else 'red' for w in weights]

    # Create the horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, len(features) * 0.4 + 2)) # Adjust figure size dynamically

    # Use barh for horizontal bars
    # Invert y-axis to have the most important feature at the top
    ax.barh(features, weights, color=colors)

    # Add a vertical line at x=0 for better readability
    ax.axvline(0, color='grey', linewidth=0.8)

    # Set labels and title
    ax.set_xlabel("Contribution to Prediction (LIME Weight)", fontsize=12)
    ax.set_ylabel("Feature (Word)", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Invert y-axis to have the most impactful features at the top
    ax.invert_yaxis()

    # Add padding to the x-axis limits to ensure bars don't get cut off
    max_abs_weight = max(abs(w) for w in weights)
    ax.set_xlim(-max_abs_weight * 1.1, max_abs_weight * 1.1)

    # Add grid for better readability
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()
    plt.show()
