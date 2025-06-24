import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode(text, word2idx, max_len=200):
    tokens = text.split()
    idxs = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    if len(idxs) < max_len:
        idxs += [word2idx['<PAD>']] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return idxs


def build_vocab(texts, min_freq=2):
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {word for word, freq in counter.items() if freq >= min_freq}
    word2idx: Dict[str, int] = {word: idx + 2 for idx, word in enumerate(sorted(vocab))}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    idx2word: Dict[int, str] = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


def cnn_tokenizer(text, word2idx, max_len, idx2word=None):
    idxs = [word2idx.get(token, word2idx['<UNK>']) for token in text.split()]
    if len(idxs) < max_len:
        idxs += [word2idx['<PAD>']] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]

    # Capture idx2word in the scope of the decode function
    _idx2word = idx2word  # Create a local reference

    def decode(idxs):
        return ' '.join([_idx2word.get(idx, '<UNK>') for idx in idxs])

    return idxs, decode


def get_pad_token_id(word2idx):
    """
    Returns the token ID for the PAD token.
    Args:
        word2idx (dict): Mapping from word to index.
    Returns:
        int: The ID of the PAD token (typically 0).
    """
    return word2idx.get('<PAD>', 0)


def create_background_texts(
        train_df: pd.DataFrame,
        text_column: str = 'text',
        label_column: int = 'label',
        n_background: int = 100,
        strategy: str = 'balanced'
) -> List[str]:
    """
    Create background texts from your training dataframe

    Args:
        train_df: Training dataframe
        text_column: Name of the column containing text data
        label_column: Name of the column containing labels (0=ham, 1=spam)
        n_background: Number of background samples to select
        strategy: How to sample ('balanced', 'random', 'stratified')

    Returns:
        List of background text samples
    """

    if strategy == 'balanced':
        # Get equal numbers of spam and ham emails
        spam_texts = train_df[train_df[label_column] == 1][text_column].tolist()
        ham_texts = train_df[train_df[label_column] == 0][text_column].tolist()

        n_spam = min(n_background // 2, len(spam_texts))
        n_ham = min(n_background // 2, len(ham_texts))

        # Randomly sample from each class
        selected_spam = np.random.choice(spam_texts, size=n_spam, replace=False)
        selected_ham = np.random.choice(ham_texts, size=n_ham, replace=False)

        background_texts = list(selected_spam) + list(selected_ham)

        print(f"Selected {n_spam} spam and {n_ham} ham emails for background")

    elif strategy == 'stratified':
        # Maintain the original class distribution
        spam_ratio = train_df[label_column].mean()
        n_spam = int(n_background * spam_ratio)
        n_ham = n_background - n_spam

        spam_texts = train_df[train_df[label_column] == 1][text_column].tolist()
        ham_texts = train_df[train_df[label_column] == 0][text_column].tolist()

        n_spam = min(n_spam, len(spam_texts))
        n_ham = min(n_ham, len(ham_texts))

        selected_spam = np.random.choice(spam_texts, size=n_spam, replace=False)
        selected_ham = np.random.choice(ham_texts, size=n_ham, replace=False)

        background_texts = list(selected_spam) + list(selected_ham)

        print(f"Selected {n_spam} spam and {n_ham} ham emails (stratified)")

    elif strategy == 'random':
        # Completely random sampling
        all_texts = train_df[text_column].tolist()
        n_samples = min(n_background, len(all_texts))
        background_texts = np.random.choice(all_texts, size=n_samples, replace=False).tolist()

        print(f"Selected {n_samples} random emails for background")

    else:
        raise ValueError("Strategy must be 'balanced', 'stratified', or 'random'")

    return background_texts


def create_diverse_background_texts(
        train_df: pd.DataFrame,
        text_column: str = 'text',
        label_column: int = 'label',
        n_background: int = 100
) -> List[str]:
    """
    Create diverse background texts by selecting samples with varying lengths
    and characteristics to better represent the data distribution

    Args:
        train_df: Training dataframe
        text_column: Name of the column containing text data
        label_column: Name of the column containing labels
        n_background: Number of background samples to select

    Returns:
        List of diverse background text samples
    """

    # Calculate text lengths
    train_df = train_df.copy()
    train_df['text_length'] = train_df[text_column].str.len()

    # Create length bins
    train_df['length_bin'] = pd.qcut(train_df['text_length'], q=4, labels=['short', 'medium', 'long', 'very_long'])

    background_texts = []
    samples_per_bin = n_background // 4

    for length_bin in ['short', 'medium', 'long', 'very_long']:
        bin_data = train_df[train_df['length_bin'] == length_bin]

        if len(bin_data) > 0:
            # Get balanced samples from this length bin
            spam_in_bin = bin_data[bin_data[label_column] == 1]
            ham_in_bin = bin_data[bin_data[label_column] == 0]

            n_spam_bin = min(samples_per_bin // 2, len(spam_in_bin))
            n_ham_bin = min(samples_per_bin // 2, len(ham_in_bin))

            if n_spam_bin > 0:
                selected_spam = spam_in_bin.sample(n=n_spam_bin)[text_column].tolist()
                background_texts.extend(selected_spam)

            if n_ham_bin > 0:
                selected_ham = ham_in_bin.sample(n=n_ham_bin)[text_column].tolist()
                background_texts.extend(selected_ham)

            print(f"{length_bin} emails: {n_spam_bin} spam, {n_ham_bin} ham")

    # If we don't have enough samples, fill with random samples
    if len(background_texts) < n_background:
        remaining = n_background - len(background_texts)
        used_indices = set()

        for text in background_texts:
            idx = train_df[train_df[text_column] == text].index
            if len(idx) > 0:
                used_indices.add(idx[0])

        available_df = train_df[~train_df.index.isin(used_indices)]
        if len(available_df) >= remaining:
            additional_texts = available_df.sample(n=remaining)[text_column].tolist()
            background_texts.extend(additional_texts)

    print(f"Total background samples: {len(background_texts)}")
    return background_texts
