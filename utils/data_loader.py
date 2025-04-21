import os
import pickle
import tarfile
import urllib.request

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