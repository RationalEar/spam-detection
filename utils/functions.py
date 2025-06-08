import random

import numpy as np
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
    word2idx = {word: idx + 2 for idx, word in enumerate(sorted(vocab))}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    idx2word = {idx: word for word, idx in word2idx.items()}
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
