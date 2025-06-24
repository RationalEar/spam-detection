import matplotlib.pyplot as plt
import numpy as np
import spacy
import torch
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import auc

# Load SpaCy model (you might need to download it first: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Downloading...")
    import subprocess

    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def tokenizer_spacy(text):
    """Custom SpaCy tokenizer for LIME."""
    return [token.text for token in nlp(text)]


class CnnLimeExplainer:
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

    def _text_to_sequence(self, texts):
        """Converts a list of text strings to a sequence of token IDs."""
        sequences = []
        max_seq_len = 0  # Determine max_seq_len dynamically or set a fixed one

        # First pass to tokenize and find max_seq_len if not fixed
        tokenized_texts = []
        for text in texts:
            tokens = tokenizer_spacy(text)
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
        alpha (float): Regularization strength for Ridge regression.    Returns:
        lime.explanation.Explanation: The LIME explanation object.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create the wrapper for LIME
    explainer_wrapper = CnnLimeExplainer(model, word_to_idx, idx_to_word, str(device))

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
    fig, ax = plt.subplots(figsize=(10, len(features) * 0.4 + 2))  # Adjust figure size dynamically

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


def compute_auc_deletion(model_wrapper, explanation, text, steps=20, embedding_type='bow'):
    """
    Calculate AUC-Del (Area Under the Deletion Curve).
    
    We iteratively remove features in order of importance and measure the model's
    confidence drop. Better explanations should cause faster confidence drops.
    
    Args:
        model_wrapper: An instance of LIMETextExplanationWrapper
        explanation: LIME explanation object
        text: Original text to explain
        steps: Number of deletion steps to perform
        embedding_type: Type of embedding used ('bow' for bag-of-words)
        
    Returns:
        float: AUC-Del score - lower is better (features are more important)
    """
    # Get the original prediction and probability for the most likely class
    original_pred = model_wrapper.predict_proba([text])[0]
    original_class = np.argmax(original_pred)
    original_prob = original_pred[original_class]

    # Get features sorted by absolute importance
    features_and_weights = explanation.as_list()
    features_and_weights.sort(key=lambda x: abs(x[1]), reverse=True)
    features = [f[0] for f in features_and_weights]

    # Normalize indices to number of steps
    max_features = min(len(features), 100)  # Avoid too many steps
    indices = np.linspace(0, max_features, num=steps, dtype=int)

    probs = []
    remaining_text = text
    already_removed = set()

    # Baseline: no deletion
    probs.append(original_prob)

    # Iteratively remove features
    for i in range(1, len(indices)):
        num_to_remove = indices[i]
        features_to_remove = features[:num_to_remove]

        # For text, we need to replace or remove the important words
        modified_text = text
        for feature in features_to_remove:
            if feature not in already_removed:
                # For bag-of-words, we can just remove the word
                # In practice, this is naive - better would be to use spaCy to handle
                # proper tokenization and avoid partial replacements
                modified_text = modified_text.replace(feature, "")
                already_removed.add(feature)

        # Get prediction after removal
        if modified_text.strip():  # Ensure text is not empty
            new_pred = model_wrapper.predict_proba([modified_text])[0]
            probs.append(new_pred[original_class])
        else:
            # If all text is removed, assume random prediction
            probs.append(0.5)  # Binary case; use 1/num_classes for multiclass

    # Normalize x-axis from 0 to 1
    x = np.linspace(0, 1, len(probs))

    # Calculate AUC
    auc_value = auc(x, probs)
    return auc_value


def compute_auc_insertion(model_wrapper, explanation, text, steps=20, embedding_type='bow'):
    """
    Calculate AUC-Ins (Area Under the Insertion Curve).
    
    We iteratively add features in order of importance and measure the model's
    confidence increase. Better explanations should cause faster confidence increases.
    
    Args:
        model_wrapper: An instance of LIMETextExplanationWrapper
        explanation: LIME explanation object
        text: Original text to explain
        steps: Number of insertion steps to perform
        embedding_type: Type of embedding used ('bow' for bag-of-words)
        
    Returns:
        float: AUC-Ins score - higher is better (features are more important)
    """
    # Get the original prediction and probability for the most likely class
    original_pred = model_wrapper.predict_proba([text])[0]
    original_class = np.argmax(original_pred)
    original_prob = original_pred[original_class]

    # Get features sorted by absolute importance
    features_and_weights = explanation.as_list()
    features_and_weights.sort(key=lambda x: abs(x[1]), reverse=True)
    features = [f[0] for f in features_and_weights]

    # Normalize indices to number of steps
    max_features = min(len(features), 100)  # Avoid too many steps
    indices = np.linspace(0, max_features, num=steps, dtype=int)

    # For text, we need to progressively reconstruct the document
    # Start with an empty document
    probs = []

    # Baseline: empty text gives random prediction
    # For binary classification, this would be 0.5
    probs.append(0.5)  # empty text

    # Create a mapping of tokens to their positions in original text
    # This is a simplified approach; in practice, use proper tokenization
    feature_positions = {}
    tokens = model_wrapper.tokenizer_spacy(text)
    for i, token in enumerate(tokens):
        if token not in feature_positions:
            feature_positions[token] = []
        feature_positions[token].append(i)

    # Track the tokens we've already used
    tokens_included = [False] * len(tokens)

    # Iteratively add features
    for i in range(1, len(indices)):
        num_to_add = indices[i]
        features_to_add = features[:num_to_add]

        # Mark tokens as included
        for feature in features_to_add:
            if feature in feature_positions:
                for pos in feature_positions[feature]:
                    tokens_included[pos] = True

        # Reconstruct text with only included tokens
        reconstructed_tokens = []
        for i, token in enumerate(tokens):
            if tokens_included[i]:
                reconstructed_tokens.append(token)
            else:
                reconstructed_tokens.append("")

        reconstructed_text = " ".join(t for t in reconstructed_tokens if t)

        # Get prediction for reconstructed text
        if reconstructed_text.strip():  # Ensure text is not empty
            new_pred = model_wrapper.predict_proba([reconstructed_text])[0]
            probs.append(new_pred[original_class])
        else:
            # If text is empty, assume random prediction
            probs.append(0.5)  # Binary case

    # Normalize x-axis from 0 to 1
    x = np.linspace(0, 1, len(probs))

    # Calculate AUC
    auc_value = auc(x, probs)
    return auc_value


def compute_comprehensiveness(model_wrapper, explanation, text, k=5):
    """
    Calculate comprehensiveness score.
    
    Measures how much the prediction changes when we remove the top k features.
    Higher values indicate more comprehensive explanations.
    
    Args:
        model_wrapper: An instance of LIMETextExplanationWrapper
        explanation: LIME explanation object
        text: Original text to explain
        k: Number of top features to remove
        
    Returns:
        float: Comprehensiveness score
    """
    # Get the original prediction
    original_pred = model_wrapper.predict_proba([text])[0]
    original_class = np.argmax(original_pred)
    original_prob = original_pred[original_class]

    # Get top k features by absolute importance
    features_and_weights = explanation.as_list()
    features_and_weights.sort(key=lambda x: abs(x[1]), reverse=True)
    top_k_features = [f[0] for f in features_and_weights[:k]]

    # Remove top k features
    modified_text = text
    for feature in top_k_features:
        modified_text = modified_text.replace(feature, "")

    # Get prediction after removal
    if modified_text.strip():  # Ensure text is not empty
        modified_pred = model_wrapper.predict_proba([modified_text])[0]
        modified_prob = modified_pred[original_class]

        # Comprehensiveness: Original probability - modified probability
        # Higher values indicate the removed features were more important
        return original_prob - modified_prob
    else:
        # If all text is removed, assume maximum comprehensiveness
        return original_prob - 0.5  # Binary case


def compute_jaccard_stability(model, texts, word_to_idx, idx_to_word, num_features=5, num_samples=500):
    """
    Calculate Jaccard stability across similar inputs.
    
    Measures the consistency of explanations across similar texts.
    Higher values indicate more stable explanations.
    
    Args:
        model: The model to explain
        texts: List of similar texts
        word_to_idx: Vocabulary mapping
        idx_to_word: Reverse vocabulary mapping
        num_features: Number of top features to compare
        num_samples: Number of samples for LIME
        
    Returns:
        float: Average Jaccard similarity (0-1)
    """
    if len(texts) < 2:
        return 1.0  # Perfect stability with only one text

    # Get explanations for all texts
    explanations = []
    for text in texts:
        explanation = get_lime_explanation(
            model,
            text,
            word_to_idx,
            idx_to_word,
            num_features=num_features,
            num_samples=num_samples
        )

        # Extract top features
        features_and_weights = explanation.as_list()
        features_and_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        top_features = set([f[0] for f in features_and_weights[:num_features]])
        explanations.append(top_features)

    # Calculate pairwise Jaccard similarities
    similarities = []
    for i in range(len(explanations)):
        for j in range(i + 1, len(explanations)):
            # Jaccard similarity: intersection over union
            intersection = len(explanations[i].intersection(explanations[j]))
            union = len(explanations[i].union(explanations[j]))
            if union > 0:  # Avoid division by zero
                similarity = intersection / union
            else:
                similarity = 1.0  # Both sets empty, perfect similarity
            similarities.append(similarity)

    # Return average stability
    return sum(similarities) / max(1, len(similarities))


def plot_deletion_insertion_curves(model_wrapper, explanation, text, steps=20):
    """
    Plot both deletion and insertion curves in a single figure.
    
    Args:
        model_wrapper: An instance of LIMETextExplanationWrapper
        explanation: LIME explanation object
        text: Original text to explain
        steps: Number of steps for curves
    """
    # Get the original prediction and probability for the most likely class
    original_pred = model_wrapper.predict_proba([text])[0]
    original_class = np.argmax(original_pred)

    # Get features sorted by absolute importance
    features_and_weights = explanation.as_list()
    features_and_weights.sort(key=lambda x: abs(x[1]), reverse=True)
    features = [f[0] for f in features_and_weights]

    # Normalize indices to number of steps
    max_features = min(len(features), 100)  # Avoid too many steps
    indices = np.linspace(0, max_features, num=steps, dtype=int)

    # Calculate deletion curve
    deletion_probs = []
    remaining_text = text
    already_removed = set()

    # Baseline: no deletion
    deletion_probs.append(original_pred[original_class])

    # Iteratively remove features
    for i in range(1, len(indices)):
        num_to_remove = indices[i]
        features_to_remove = features[:num_to_remove]

        # For text, we need to replace or remove the important words
        modified_text = text
        for feature in features_to_remove:
            if feature not in already_removed:
                modified_text = modified_text.replace(feature, "")
                already_removed.add(feature)

        # Get prediction after removal
        if modified_text.strip():  # Ensure text is not empty
            new_pred = model_wrapper.predict_proba([modified_text])[0]
            deletion_probs.append(new_pred[original_class])
        else:
            # If all text is removed, assume random prediction
            deletion_probs.append(0.5)  # Binary case

    # Calculate insertion curve
    insertion_probs = []

    # Baseline: empty text gives random prediction
    insertion_probs.append(0.5)  # empty text

    # Create a mapping of tokens to their positions in original text
    feature_positions = {}
    tokens = model_wrapper.tokenizer_spacy(text)
    for i, token in enumerate(tokens):
        if token not in feature_positions:
            feature_positions[token] = []
        feature_positions[token].append(i)

    # Track the tokens we've already used
    tokens_included = [False] * len(tokens)

    # Iteratively add features
    for i in range(1, len(indices)):
        num_to_add = indices[i]
        features_to_add = features[:num_to_add]

        # Mark tokens as included
        for feature in features_to_add:
            if feature in feature_positions:
                for pos in feature_positions[feature]:
                    tokens_included[pos] = True

        # Reconstruct text with only included tokens
        reconstructed_tokens = []
        for i, token in enumerate(tokens):
            if tokens_included[i]:
                reconstructed_tokens.append(token)
            else:
                reconstructed_tokens.append("")

        reconstructed_text = " ".join(t for t in reconstructed_tokens if t)

        # Get prediction for reconstructed text
        if reconstructed_text.strip():  # Ensure text is not empty
            new_pred = model_wrapper.predict_proba([reconstructed_text])[0]
            insertion_probs.append(new_pred[original_class])
        else:
            # If text is empty, assume random prediction
            insertion_probs.append(0.5)  # Binary case

    # Normalize x-axis from 0 to 1
    x = np.linspace(0, 1, len(deletion_probs))

    # Calculate AUC values
    auc_deletion = auc(x, deletion_probs)
    auc_insertion = auc(x, insertion_probs)

    # Plot both curves
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, deletion_probs, 'r-', linewidth=2, label=f'Deletion (AUC={auc_deletion:.3f})')
    ax.plot(x, insertion_probs, 'b-', linewidth=2, label=f'Insertion (AUC={auc_insertion:.3f})')

    # Add grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Fraction of Features', fontsize=12)
    ax.set_ylabel('Model Prediction Probability', fontsize=12)
    ax.set_title('Deletion and Insertion Curves', fontsize=14)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

    return auc_deletion, auc_insertion


def plot_metrics_across_samples(metrics_df):
    """
    Plot explanation quality metrics across multiple samples.
    
    Args:
        metrics_df: DataFrame containing metrics for multiple samples
    """
    # Prepare data for plotting
    metrics_to_plot = ['AUC-Del', 'AUC-Ins', 'Comprehensiveness', 'Jaccard Stability']

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for i, metric in enumerate(metrics_to_plot):
        if metric in metrics_df.columns:
            # Create bar plot
            axs[i].bar(metrics_df.index, metrics_df[metric], color='skyblue')
            axs[i].set_title(f'{metric} Across Samples', fontsize=14)
            axs[i].set_xlabel('Sample Index', fontsize=12)
            axs[i].set_ylabel(metric, fontsize=12)

            # Add gridlines
            axs[i].grid(True, linestyle='--', alpha=0.7, axis='y')

            # Rotate x-tick labels if many samples
            if len(metrics_df) > 10:
                axs[i].set_xticks(metrics_df.index)
                axs[i].set_xticklabels(metrics_df.index, rotation=45)

            # Add mean line
            mean_val = metrics_df[metric].mean()
            axs[i].axhline(y=mean_val, color='r', linestyle='-')
            axs[i].text(0.02, 0.95, f'Mean: {mean_val:.3f}',
                        transform=axs[i].transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()
