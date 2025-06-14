import hashlib
import os
import pickle
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

from preprocess.email_parser import parse_email
from utils.constants import DATA_PATH

# Helper: Redact PII and hash emails
EMAIL_PATTERN = re.compile(r'[\w.-]+@[\w.-]+')
CURRENCY_PATTERN = re.compile(r'\$\d+(?:\.\d+)?')
URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+')
PHONE_PATTERN = re.compile(r'\b\d{10,}\b')
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s<>]')
WHITESPACE_PATTERN = re.compile(r'\s+')
STOP_WORDS = {"a", "an", "the", "and", "or", "but", "if", "while", "with", "of", "at", "by", "for", "to", "in", "on",
              "is", "it", "this", "that", "as", "are", "was", "were", "be", "been", "has", "have", "had", "do", "does",
              "did", "from", "so", "such", "can", "into", "than", "then", "too", "also", "out", "up"}


def redact_and_hash_email(text):
    def hash_email(match):
        return hashlib.sha256(match.group(0).encode()).hexdigest()

    text = EMAIL_PATTERN.sub(hash_email, text)
    return text


def preprocess_text(raw_email):
    if not isinstance(raw_email, str):
        return ""

    # Remove email headers/footers
    raw_email = re.sub(r"^.*?(From:|Subject:|To:).*?\n", "", raw_email, flags=re.DOTALL | re.IGNORECASE)
    raw_email = re.sub(r"\n\S+:\s.*?\n", "\n", raw_email)  # Remove remaining headers

    # Normalization
    raw_email = raw_email.lower()

    # Strip HTML
    soup = BeautifulSoup(raw_email, 'html.parser')
    text = soup.get_text()

    text = URL_PATTERN.sub('<URL>', text)  # Replace URLs
    text = EMAIL_PATTERN.sub("<EMAIL>", text)  # Replace emails in the body
    text = PHONE_PATTERN.sub("<PHONE>", text)  # Replace phone numbers
    text = CURRENCY_PATTERN.sub('<CURRENCY>', text)  # Replace currencies
    text = SPECIAL_CHARS_PATTERN.sub(" ", text)  # Replace special chars with space
    text = WHITESPACE_PATTERN.sub(" ", text).strip()  # Collapse whitespace

    # Remove stop words
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    text = " ".join(tokens)

    return text


def hash_email_address(email_addr):
    return hashlib.sha256(email_addr.encode()).hexdigest()


def create_dataset():
    data = []
    email_hash_dict = {}

    # Process each dataset with proper labeling
    for dataset, label in [("easy_ham", 0), ("easy_ham_2", 0),
                           ("hard_ham", 0), ("spam", 1), ("spam_2", 1)]:
        dir_path = f"{DATA_PATH}/data/raw/{dataset}"
        if not os.path.exists(dir_path):
            continue

        for filename in os.listdir(dir_path):
            if filename.startswith(".") or "cmds" in filename:
                continue

            file_path = os.path.join(dir_path, filename)
            parsed = parse_email(file_path)
            if parsed:
                # Hash sender and reply-to emails, store mapping
                sender_email = parsed.get("sender")
                reply_to_email = parsed.get("reply_to")
                sender_hash = hash_email_address(sender_email) if sender_email else ""
                reply_to_hash = hash_email_address(reply_to_email) if reply_to_email else ""
                if sender_email:
                    email_hash_dict[sender_email] = sender_hash
                if reply_to_email:
                    email_hash_dict[reply_to_email] = reply_to_hash

                raw_email = f"<SUBJECT>{parsed['subject']}</SUBJECT> <BODY>{parsed['body']}</BODY>"
                data.append({
                    "subject": parsed["subject"],
                    "text": preprocess_text(raw_email),
                    "label": label,
                    "source": dataset,  # Track origin
                    "sender_hash": sender_hash,
                    "reply_to_hash": reply_to_hash,
                    "date": parsed.get("date", "")  # Store the date for temporal split
                })
    df = pd.DataFrame(data)
    df = df.reset_index(drop=True)
    return df, email_hash_dict


def parse_email_date(date_str):
    # Try to parse the date string into a sortable offset-naive UTC datetime object
    from email.utils import parsedate_to_datetime
    import pandas as pd
    try:
        dt = parsedate_to_datetime(date_str)
        if dt is None:
            return pd.Timestamp.min
        # Convert to UTC and remove tzinfo to make offset-naive
        if dt.tzinfo is not None:
            dt = dt.astimezone(tz=None).replace(tzinfo=None)
        return pd.Timestamp(dt)
    except Exception:
        return pd.Timestamp.min


def temporal_stratified_split(df, stratify_col="label", date_col="date"):
    # Split each class by time (date order) into 80/10/10
    train_idx, val_idx, test_idx = [], [], []
    for label in df[stratify_col].unique():
        class_df = df[df[stratify_col] == label].copy()
        # Parse dates for sorting (all offset-naive)
        class_df["_parsed_date"] = class_df[date_col].apply(parse_email_date)
        # Sort by parsed date, fallback to index if date missing
        class_df = class_df.sort_values(by=["_parsed_date", date_col, "text"])
        n = len(class_df)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        idx = class_df.index.tolist()
        train_idx += idx[:n_train]
        val_idx += idx[n_train:n_train + n_val]
        test_idx += idx[n_train + n_val:]
    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True)
    )


def prepare_data():
    if not os.path.exists(f"{DATA_PATH}/data/processed/train.pkl"):
        df, email_hash_dict = create_dataset()

        print(f"\nDataset counts:")
        print(df["source"].value_counts())

        # Stratified temporal split 80/10/10
        train_df, val_df, test_df = temporal_stratified_split(df, stratify_col="label", date_col="date")
        print(f"Spam ratio: {df['label'].mean():.2%}")

        # Save data
        os.makedirs(f"{DATA_PATH}/data/processed", exist_ok=True)
        train_df.to_csv(f"{DATA_PATH}/data/processed/train.csv", index=False)
        val_df.to_csv(f"{DATA_PATH}/data/processed/val.csv", index=False)
        test_df.to_csv(f"{DATA_PATH}/data/processed/test.csv", index=False)

        with open(f"{DATA_PATH}/data/processed/train.pkl", "wb") as f:
            pickle.dump(train_df, f)
        with open(f"{DATA_PATH}/data/processed/val.pkl", "wb") as f:
            pickle.dump(val_df, f)
        with open(f"{DATA_PATH}/data/processed/test.pkl", "wb") as f:
            pickle.dump(test_df, f)
        with open(f"{DATA_PATH}/data/processed/email_hash_dict.pkl", "wb") as f:
            pickle.dump(email_hash_dict, f)
    else:
        print("Data already prepared. Loading from disk...")
        train_df = pd.read_csv(f"{DATA_PATH}/data/processed/train.csv")
        val_df = pd.read_csv(f"{DATA_PATH}/data/processed/val.csv")
        test_df = pd.read_csv(f"{DATA_PATH}/data/processed/test.csv")

    print("\nData preparation complete!")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}, Total: {len(train_df) + len(val_df) + len(test_df)}")