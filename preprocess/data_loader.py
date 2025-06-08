import os
import tarfile
import numpy as np
import torch
import urllib.request
from utils.constants import DATA_PATH

DATASETS = [
    ("easy_ham", "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2"),
    ("easy_ham_2", "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2"),
    ("hard_ham", "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2"),
    ("spam", "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2"),
    ("spam_2", "https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2")
]


def download_datasets():
    """
    Download and extract the datasets from the given URLs.
    """
    for dir_name, url in DATASETS:
        print(f"Downloading {dir_name}...")
        file_path = f"{DATA_PATH}/data/raw/{dir_name}.tar.bz2"
        # download if not already downloaded
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(url, file_path)
        else:
            print(f"{dir_name} already downloaded.")

        # Extract with folder structure preservation
        with tarfile.open(file_path, "r:bz2") as tar:
            tar.extractall(f"{DATA_PATH}/data/raw")


def load_glove_embeddings(glove_path, word2idx, embedding_dim=300):
    """
    Loads GloVe embeddings and creates an embedding matrix for the given vocabulary.
    Args:
        glove_path (str): Path to the GloVe .txt file.
        word2idx (dict): Mapping from word to index in your vocabulary.
        embedding_dim (int): Dimension of the embeddings (default 300).
    Returns:
        torch.Tensor: Embedding matrix of shape (vocab_size, embedding_dim)
    """
    embeddings_index = {}
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    for word, idx in word2idx.items():
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return torch.tensor(embedding_matrix)
