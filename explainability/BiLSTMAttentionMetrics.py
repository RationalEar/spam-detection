import random
from typing import List, Dict, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import auc


class BiLSTMAttentionMetrics:
    def __init__(self, model, word2idx, idx2word, max_len=200, device='cpu'):
        """
        Initialize explanation quality metrics calculator for BiLSTM attention

        Args:
            model: BiLSTMSpam model instance
            word2idx: Word to index mapping
            idx2word: Index to word mapping
            max_len: Maximum sequence length
            device: Device to run computations on
        """
        self.model = model
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.max_len = max_len
        self.device = device
        self.model.eval()

    def _encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to token indices
        
        Args:
            text: Input text
            
        Returns:
            Tensor of token indices
        """
        from utils.functions import encode
        return torch.tensor(encode(text, self.word2idx, self.max_len))

    def _get_attention_weights(self, text: str) -> np.ndarray:
        """
        Get attention weights for the given text
        
        Args:
            text: Input text
            
        Returns:
            Attention weights as numpy array
        """
        encoded_text = self._encode_text(text).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, attention_weights = self.model(encoded_text)
            
        return attention_weights.cpu().numpy()[0]  # Remove batch dimension

    def _get_token_importance_ranking(self, text: str, attention_weights: np.ndarray) -> List[Tuple[int, float]]:
        """
        Get token importance ranking from attention weights

        Args:
            text: Original text
            attention_weights: Attention weights array

        Returns:
            List of (token_index, importance_score) tuples sorted by importance
        """
        tokens = text.split()
        
        # Get importance scores for valid tokens only (exclude padding)
        token_importance = []
        for i, token in enumerate(tokens):
            if i < len(attention_weights) and i < self.max_len:
                importance = attention_weights[i]
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
        sequence = self._encode_text(text).clone()

        # Filter out indices that are out of bounds
        valid_mask_indices = [idx for idx in mask_indices if idx < len(sequence)]

        if mask_strategy == 'zero':
            # Replace with padding token (0)
            sequence[valid_mask_indices] = 0
        elif mask_strategy == 'random':
            # Replace with random tokens from vocabulary
            vocab_size = len(self.word2idx)
            random_tokens = torch.randint(1, vocab_size, (len(valid_mask_indices),))
            sequence[valid_mask_indices] = random_tokens
        elif mask_strategy == 'unk':
            # Replace with unknown token if available
            unk_idx = self.word2idx.get('<UNK>', 0)
            sequence[valid_mask_indices] = unk_idx

        return sequence

    def _get_prediction(self, sequence: torch.Tensor) -> float:
        """
        Get model prediction for a sequence
        
        Args:
            sequence: Input sequence tensor
            
        Returns:
            Prediction probability
        """
        sequence = sequence.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred, _ = self.model(sequence)
            return pred.item()

    def compute_auc_deletion(self, text: str, steps: int = 20, 
                           mask_strategy: str = 'zero') -> float:
        """
        Compute AUC-Del: Area under deletion curve
        Lower values indicate better explanations

        Args:
            text: Original text
            steps: Number of deletion steps
            mask_strategy: How to mask tokens ('zero', 'random', 'unk')

        Returns:
            AUC-Del score (lower is better)
        """
        # Get attention weights
        attention_weights = self._get_attention_weights(text)
        
        # Get original prediction
        original_seq = self._encode_text(text)
        original_pred = self._get_prediction(original_seq)

        # Get token importance ranking
        token_ranking = self._get_token_importance_ranking(text, attention_weights)

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
            masked_pred = self._get_prediction(masked_seq)

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
        # Get attention weights
        attention_weights = self._get_attention_weights(text)
        
        # Start with completely masked sequence
        tokens = text.split()
        if not tokens:
            return 0.0

        all_masked_seq = self._create_masked_sequence(text, list(range(len(tokens))), 'zero')
        baseline_pred = self._get_prediction(all_masked_seq)

        # Get token importance ranking
        token_ranking = self._get_token_importance_ranking(text, attention_weights)

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
            pred = self._get_prediction(partially_masked_seq)

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
        # Get attention weights
        attention_weights = self._get_attention_weights(text)
        
        # Get original prediction
        original_seq = self._encode_text(text)
        original_pred = self._get_prediction(original_seq)

        # Get top-k most important tokens
        token_ranking = self._get_token_importance_ranking(text, attention_weights)
        top_k_indices = [idx for idx, _ in token_ranking[:k]]

        if not top_k_indices:
            return 0.0

        # Create sequence with top-k tokens removed
        masked_seq = self._create_masked_sequence(text, top_k_indices, 'zero')
        masked_pred = self._get_prediction(masked_seq)

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
        # Get original attention
        original_attention = self._get_attention_weights(text)
        original_ranking = self._get_token_importance_ranking(text, original_attention)
        original_top_k = set([idx for idx, _ in original_ranking[:k]])

        jaccard_scores = []
        tokens = text.split()

        for _ in range(num_perturbations):
            # Create perturbed version by randomly replacing some tokens
            perturbed_tokens = tokens.copy()

            for i in range(len(perturbed_tokens)):
                if random.random() < perturbation_prob:
                    # Replace with a random token from vocabulary
                    vocab_words = [word for word in self.word2idx.keys() 
                                 if word not in ['<PAD>', '<UNK>']]
                    if vocab_words:
                        perturbed_tokens[i] = random.choice(vocab_words)

            # Reconstruct text
            perturbed_text = ' '.join(perturbed_tokens)

            try:
                # Get attention weights for perturbed text
                perturbed_attention = self._get_attention_weights(perturbed_text)
                perturbed_ranking = self._get_token_importance_ranking(perturbed_text, perturbed_attention)
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
        Compute rank correlation: Spearman's ρ between attention weights for perturbed samples
        Higher values indicate more stable explanations

        Args:
            text: Original text
            num_perturbations: Number of perturbed versions to generate
            perturbation_prob: Probability of perturbing each token

        Returns:
            Average Spearman correlation (higher is better)
        """
        # Get original attention weights
        original_attention = self._get_attention_weights(text)
        tokens = text.split()
        
        correlations = []

        for _ in range(num_perturbations):
            # Create perturbed version
            perturbed_tokens = tokens.copy()

            for i in range(len(perturbed_tokens)):
                if random.random() < perturbation_prob:
                    vocab_words = [word for word in self.word2idx.keys() 
                                 if word not in ['<PAD>', '<UNK>']]
                    if vocab_words:
                        perturbed_tokens[i] = random.choice(vocab_words)

            perturbed_text = ' '.join(perturbed_tokens)

            try:
                # Get attention weights for perturbed text
                perturbed_attention = self._get_attention_weights(perturbed_text)
                
                # Calculate correlation for overlapping positions
                min_len = min(len(original_attention), len(perturbed_attention), len(tokens))
                if min_len > 1:
                    corr, _ = spearmanr(original_attention[:min_len], 
                                      perturbed_attention[:min_len])
                    if not np.isnan(corr):
                        correlations.append(corr)

            except Exception as e:
                print(f"Error processing perturbed text: {e}")
                continue

        return np.mean(correlations) if correlations else 0.0

    def evaluate_all_metrics(self, text: str, steps: int = 20, k: int = 5,
                           num_perturbations: int = 10, perturbation_prob: float = 0.1) -> Dict[str, float]:
        """
        Compute all explanation quality metrics for a given text

        Args:
            text: Input text
            steps: Number of steps for AUC calculations
            k: Number of top features for comprehensiveness and Jaccard stability
            num_perturbations: Number of perturbations for stability metrics
            perturbation_prob: Probability of perturbing each token

        Returns:
            Dictionary with all metric scores
        """
        print(f"Evaluating metrics for text: {text[:50]}...")
        
        results = {}
        
        try:
            results['auc_deletion'] = self.compute_auc_deletion(text, steps)
            print(f"AUC-Del: {results['auc_deletion']:.4f}")
        except Exception as e:
            print(f"Error computing AUC-Del: {e}")
            results['auc_deletion'] = 0.0

        try:
            results['auc_insertion'] = self.compute_auc_insertion(text, steps)
            print(f"AUC-Ins: {results['auc_insertion']:.4f}")
        except Exception as e:
            print(f"Error computing AUC-Ins: {e}")
            results['auc_insertion'] = 0.0

        try:
            results['comprehensiveness'] = self.compute_comprehensiveness(text, k)
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

    def visualize_attention(self, text: str, save_path: str = None) -> None:
        """
        Visualize attention weights for a given text

        Args:
            text: Input text
            save_path: Path to save the visualization (optional)
        """
        attention_weights = self._get_attention_weights(text)
        tokens = text.split()
        
        # Limit to actual tokens (exclude padding)
        num_tokens = min(len(tokens), len(attention_weights))
        tokens = tokens[:num_tokens]
        weights = attention_weights[:num_tokens]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(tokens)), weights)
        plt.xlabel('Tokens')
        plt.ylabel('Attention Weight')
        plt.title('BiLSTM Attention Weights')
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        
        # Color bars based on attention weight
        max_weight = max(weights) if weights else 1
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.Blues(weights[i] / max_weight))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    # Calculate explanation quality metrics for the entire test dataset (or a subset)
    def calculate_overall_metrics(self, test_df, sample_size=100):
        # Sample a subset if the test set is large
        if len(test_df) > sample_size:
            sample_indices = np.random.choice(len(test_df), sample_size, replace=False)
            eval_df = test_df.iloc[sample_indices]
        else:
            eval_df = test_df

        all_metrics = []

        for i, row in enumerate(eval_df.itertuples()):
            print(f"Processing example {i + 1}/{len(eval_df)}", end="\r")
            text = row.text

            # Skip texts that are too short
            if len(text.split()) < 3:
                continue

            metrics = self.evaluate_all_metrics(
                text=text,
                steps=10,  # Reduced for efficiency
                k=5,
                num_perturbations=5,
                perturbation_prob=0.1
            )

            # Add label and prediction
            metrics['true_label'] = row.label
            with torch.no_grad():
                input_tensor = self._encode_text(text).unsqueeze(0).to(self.device)
                pred = self.model(input_tensor)
                if isinstance(pred, tuple):
                    pred = pred[0]
                metrics['predicted_prob'] = pred.item()

            all_metrics.append(metrics)

        # Convert to dataframe
        metrics_df = pd.DataFrame(all_metrics)

        # Calculate overall metrics
        overall_metrics = {
            'mean': metrics_df.mean(),
            'median': metrics_df.median(),
            'std': metrics_df.std()
        }

        # Calculate metrics separately for spam and ham
        spam_metrics = metrics_df[metrics_df['true_label'] == 1].mean()
        ham_metrics = metrics_df[metrics_df['true_label'] == 0].mean()

        return metrics_df, overall_metrics, spam_metrics, ham_metrics

    def get_top_influential_words(self, text: str, top_k: int = 20) -> List[Dict[str, any]]:
        """
        Get top-k most influential words/tokens for a single text using attention weights

        Args:
            text: Input text
            top_k: Number of top influential tokens to return

        Returns:
            List of dictionaries with token info and importance scores
        """
        # Get attention weights
        attention_weights = self._get_attention_weights(text)
        tokens = text.split()

        # Get token importance ranking
        token_ranking = self._get_token_importance_ranking(text, attention_weights)

        # Build result list with token information
        influential_words = []
        for rank, (token_idx, importance_score) in enumerate(token_ranking[:top_k]):
            if token_idx < len(tokens):
                token = tokens[token_idx]

                influential_words.append({
                    'rank': rank + 1,
                    'token': token,
                    'word': token.lower(),
                    'token_index': token_idx,
                    'importance_score': importance_score
                })

        return influential_words[:top_k]

    def analyze_dataset_influential_words(self, texts: List[str], labels: List[int] = None,
                                        top_k: int = 20, verbose: bool = True) -> Dict[str, any]:
        """
        Analyze top influential words across entire dataset using attention weights

        Args:
            texts: List of input texts
            labels: Optional list of labels (0=ham, 1=spam)
            top_k: Number of top words to analyze per text
            verbose: Whether to print progress

        Returns:
            Dictionary with comprehensive analysis results
        """
        if verbose:
            print(f"Analyzing {len(texts)} texts for influential words using BiLSTM attention...")

        all_word_scores = {}  # word -> list of scores
        spam_word_scores = {}  # word -> list of scores (spam only)
        ham_word_scores = {}   # word -> list of scores (ham only)

        processed_count = 0

        for i, text in enumerate(texts):
            try:
                # Skip very short texts
                if len(text.split()) < 3:
                    continue

                # Get top influential words for this text
                influential_words = self.get_top_influential_words(text, top_k)

                # Determine label
                is_spam = labels[i] == 1 if labels and i < len(labels) else None

                # Aggregate scores
                for word_info in influential_words:
                    word = word_info['word'].lower()
                    score = word_info['importance_score']

                    # Overall scores
                    if word not in all_word_scores:
                        all_word_scores[word] = []
                    all_word_scores[word].append(score)

                    # Label-specific scores
                    if is_spam is not None:
                        if is_spam:
                            if word not in spam_word_scores:
                                spam_word_scores[word] = []
                            spam_word_scores[word].append(score)
                        else:
                            if word not in ham_word_scores:
                                ham_word_scores[word] = []
                            ham_word_scores[word].append(score)

                processed_count += 1

                if verbose and (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(texts)} texts...")

            except Exception as e:
                if verbose:
                    print(f"Error processing text {i + 1}: {e}")
                continue

        if verbose:
            print(f"Successfully processed {processed_count} texts")

        # Calculate aggregated statistics
        def calculate_word_stats(word_scores_dict):
            word_stats = []
            for word, scores in word_scores_dict.items():
                if len(scores) >= 2:  # Only include words that appear multiple times
                    word_stats.append({
                        'word': word,
                        'frequency': len(scores),
                        'mean_importance': np.mean(scores),
                        'std_importance': np.std(scores),
                        'max_importance': np.max(scores),
                        'total_importance': np.sum(scores)
                    })

            # Sort by mean importance
            word_stats.sort(key=lambda x: x['mean_importance'], reverse=True)
            return word_stats

        # Calculate statistics for all categories
        overall_stats = calculate_word_stats(all_word_scores)
        spam_stats = calculate_word_stats(spam_word_scores) if spam_word_scores else []
        ham_stats = calculate_word_stats(ham_word_scores) if ham_word_scores else []

        # Find discriminative words (appear more in one class than the other)
        discriminative_words = []
        if spam_stats and ham_stats:
            spam_words = {w['word']: w['mean_importance'] for w in spam_stats}
            ham_words = {w['word']: w['mean_importance'] for w in ham_stats}

            all_words = set(spam_words.keys()) | set(ham_words.keys())

            for word in all_words:
                spam_score = spam_words.get(word, 0)
                ham_score = ham_words.get(word, 0)

                if spam_score > 0 or ham_score > 0:
                    discriminative_score = spam_score - ham_score
                    discriminative_words.append({
                        'word': word,
                        'spam_importance': spam_score,
                        'ham_importance': ham_score,
                        'discriminative_score': discriminative_score,
                        'category': 'spam_indicator' if discriminative_score > 0 else 'ham_indicator'
                    })

            discriminative_words.sort(key=lambda x: abs(x['discriminative_score']), reverse=True)

        return {
            'method': 'bilstm_attention',
            'total_texts_processed': processed_count,
            'top_overall_words': overall_stats[:top_k],
            'top_spam_words': spam_stats[:top_k],
            'top_ham_words': ham_stats[:top_k],
            'top_discriminative_words': discriminative_words[:top_k],
            'statistics': {
                'total_unique_words': len(all_word_scores),
                'spam_unique_words': len(spam_word_scores),
                'ham_unique_words': len(ham_word_scores)
            }
        }

    def print_influential_words_analysis(self, analysis_results: Dict[str, any]):
        """
        Print formatted analysis results for influential words

        Args:
            analysis_results: Results from analyze_dataset_influential_words
        """
        print("\n" + "=" * 80)
        print("TOP INFLUENTIAL WORDS ANALYSIS - BiLSTM ATTENTION")
        print("=" * 80)
        print(f"Method: {analysis_results['method']}")
        print(f"Total texts processed: {analysis_results['total_texts_processed']}")
        print(f"Total unique words found: {analysis_results['statistics']['total_unique_words']}")

        # Overall top words
        print(f"\n{'='*50}")
        print("TOP 20 MOST INFLUENTIAL WORDS (OVERALL)")
        print(f"{'='*50}")
        print(f"{'Rank':<4} {'Word':<20} {'Frequency':<10} {'Mean Score':<12} {'Max Score':<10}")
        print("-" * 60)

        for i, word_info in enumerate(analysis_results['top_overall_words'][:20]):
            print(f"{i+1:<4} {word_info['word']:<20} {word_info['frequency']:<10} "
                  f"{word_info['mean_importance']:<12.4f} {word_info['max_importance']:<10.4f}")

        # Spam-specific words
        if analysis_results['top_spam_words']:
            print(f"\n{'='*50}")
            print("TOP 20 MOST INFLUENTIAL WORDS (SPAM)")
            print(f"{'='*50}")
            print(f"{'Rank':<4} {'Word':<20} {'Frequency':<10} {'Mean Score':<12} {'Max Score':<10}")
            print("-" * 60)

            for i, word_info in enumerate(analysis_results['top_spam_words'][:20]):
                print(f"{i+1:<4} {word_info['word']:<20} {word_info['frequency']:<10} "
                      f"{word_info['mean_importance']:<12.4f} {word_info['max_importance']:<10.4f}")

        # Ham-specific words
        if analysis_results['top_ham_words']:
            print(f"\n{'='*50}")
            print("TOP 20 MOST INFLUENTIAL WORDS (HAM)")
            print(f"{'='*50}")
            print(f"{'Rank':<4} {'Word':<20} {'Frequency':<10} {'Mean Score':<12} {'Max Score':<10}")
            print("-" * 60)

            for i, word_info in enumerate(analysis_results['top_ham_words'][:20]):
                print(f"{i+1:<4} {word_info['word']:<20} {word_info['frequency']:<10} "
                      f"{word_info['mean_importance']:<12.4f} {word_info['max_importance']:<10.4f}")

        # Discriminative words
        if analysis_results['top_discriminative_words']:
            print(f"\n{'='*50}")
            print("TOP 20 MOST DISCRIMINATIVE WORDS")
            print(f"{'='*50}")
            print(f"{'Rank':<4} {'Word':<20} {'Category':<15} {'Spam Score':<12} {'Ham Score':<11} {'Diff':<8}")
            print("-" * 75)

            for i, word_info in enumerate(analysis_results['top_discriminative_words'][:20]):
                print(f"{i+1:<4} {word_info['word']:<20} {word_info['category']:<15} "
                      f"{word_info['spam_importance']:<12.4f} {word_info['ham_importance']:<11.4f} "
                      f"{word_info['discriminative_score']:<8.4f}")

        print("\n" + "=" * 80)


def analyze_test_dataset_influential_words_bilstm(model, word2idx, idx2word, test_texts: List[str],
                                                test_labels: List[int] = None, device: str = 'cpu',
                                                max_len: int = 200, top_k: int = 20):
    """
    Standalone function to analyze influential words in test dataset using BiLSTM attention

    Args:
        model: Trained BiLSTM model
        word2idx: Word to index mapping
        idx2word: Index to word mapping
        test_texts: List of test texts
        test_labels: Optional list of test labels
        device: Device to run on
        max_len: Maximum sequence length
        top_k: Number of top words to analyze

    Returns:
        Dictionary with analysis results
    """
    print("Initializing BiLSTM attention analyzer...")
    analyzer = BiLSTMAttentionMetrics(model, word2idx, idx2word, max_len, device)

    print(f"Analyzing top {top_k} influential words using BiLSTM attention...")
    results = analyzer.analyze_dataset_influential_words(
        texts=test_texts,
        labels=test_labels,
        top_k=top_k,
        verbose=True
    )

    # Print formatted results
    analyzer.print_influential_words_analysis(results)

    return results

