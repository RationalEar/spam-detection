import os
import pickle
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.data_loader import download_datasets
from utils.email_parser import parse_email


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove email headers/footers
    text = re.sub(r"^.*?(From:|Subject:|To:).*?\n", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n\S+:\s.*?\n", "\n", text)  # Remove remaining headers
    
    # Normalization
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "<URL>", text)  # Replace URLs
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)  # Replace emails
    text = re.sub(r"\b\d{10,}\b", "<PHONE>", text)  # Replace phone numbers
    text = re.sub(r"[^\w\s<>]", " ", text)  # Replace special chars with space
    text = re.sub(r"\s+", " ", text).strip()  # Collapse whitespace
    return text


def create_dataset():
    data = []
    
    # Process each dataset with proper labeling
    for dataset, label in [("easy_ham", 0), ("easy_ham_2", 0),
                           ("hard_ham", 0), ("spam", 1), ("spam_2", 1)]:
        dir_path = f"data/raw/{dataset}"
        if not os.path.exists(dir_path):
            continue
        
        for filename in os.listdir(dir_path):
            if filename.startswith(".") or "cmds" in filename:
                continue
            
            file_path = os.path.join(dir_path, filename)
            parsed = parse_email(file_path)
            if parsed:
                text = f"{parsed['subject']} {parsed['body']}"
                data.append({
                    "text": preprocess_text(text),
                    "label": label,
                    "source": dataset  # Track origin
                })
    
    return pd.DataFrame(data)


def prepare_data():
    download_datasets()
    df = create_dataset()
    
    # Verify counts match original description
    print(f"\nDataset counts:")
    print(df["source"].value_counts())
    
    # Stratified split (preserve class balance)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )
    
    # Save data
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    with open("data/processed/train.pkl", "wb") as f:
        pickle.dump(train_df, f)
    with open("data/processed/test.pkl", "wb") as f:
        pickle.dump(test_df, f)
    
    print("\nData preparation complete!")
    print(f"Total samples: {len(df)}")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Spam ratio: {df['label'].mean():.2%}")