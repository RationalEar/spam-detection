import os
import pickle
import tarfile
import urllib.request

import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from utils.EmailDataset import EmailDataset

datasets = [
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
    for dir_name, url in datasets:
        print(f"Downloading {dir_name}...")
        file_path = f"data/raw/{dir_name}.tar.bz2"
        # download if not already downloaded
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(url, file_path)
        else:
            print(f"{dir_name} already downloaded.")
        
        # Extract with folder structure preservation
        with tarfile.open(file_path, "r:bz2") as tar:
            tar.extractall("data/raw")


def load_data():
    # Load from pickle (faster)
    with open("data/train.pkl", "rb") as f:
        train_df = pickle.load(f)
    with open("data/test.pkl", "rb") as f:
        test_df = pickle.load(f)
    
    X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    X_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()
    return X_train, X_test, y_train, y_test


def load_datasets(tokenizer):
    # Load preprocessed data (from Section 3.2)
    train_df = pd.read_pickle('data/processed/train.pkl')
    test_df = pd.read_pickle('data/processed/test.pkl')
    
    # Create datasets
    train_dataset = EmailDataset(train_df['text'], train_df['label'], tokenizer)
    test_dataset = EmailDataset(test_df['text'], test_df['label'], tokenizer)
    
    return train_dataset, test_dataset


def get_test_loader(batch_size=32, shuffle=False, dir_root=''):
    """Create test DataLoader with consistent parameters"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_df = pd.read_pickle(dir_root + 'data/processed/test.pkl')
    test_dataset = EmailDataset(test_df['text'], test_df['label'], tokenizer)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)


def get_train_loader(batch_size=32, shuffle=True, dir_root=''):
    """Create training DataLoader with consistent parameters"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_df = pd.read_pickle(dir_root + 'data/processed/train.pkl')
    train_dataset = EmailDataset(train_df['text'], train_df['label'], tokenizer)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
