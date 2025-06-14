import os

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

from training.bert_trainer import train_bert
from training.bilstm_trainer import train_bilstm
from training.cnn_trainer import train_cnn
from utils.functions import build_vocab, encode


def train_model(model_type, train_df, val_df=None, test_df=None, embedding_dim=300, pretrained_embeddings=None,
                model_save_path='', max_len=200, evaluate=False, early_stopping_patience=5):
    """
    Train a model with support for 80/10/10 train/val/test split
    
    Args:
        model_type: Type of model to train ('cnn', 'bilstm', or 'bert')
        train_df: DataFrame containing training data
        val_df: DataFrame containing validation data (if None, will use test_df for validation)
        test_df: DataFrame containing test data
        embedding_dim: Dimension of word embeddings
        pretrained_embeddings: Pre-trained embeddings tensor
        model_save_path: Directory to save model checkpoints
        max_len: Maximum sequence length
        evaluate: Whether to evaluate the model after training
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped
    """
    # Check if we have all three splits
    has_three_way_split = val_df is not None and test_df is not None
    
    if not has_three_way_split:
        if test_df is None:
            raise ValueError("test_df must be provided")
        # Fall back to two-way split if validation set not provided
        print("Warning: No validation set provided. Using test set for validation.")
        val_df = test_df
    
    # Create save directory if it doesn't exist
    if model_save_path:
        os.makedirs(model_save_path, exist_ok=True)
    
    # Choose model: 'cnn', 'bilstm', or 'bert'
    if model_type == 'cnn':
        model = train_cnn(train_df, val_df, test_df, embedding_dim, pretrained_embeddings, 
                         model_save_path, max_len, early_stopping_patience)
    elif model_type == 'bilstm':
        model = train_bilstm(train_df, val_df, test_df, embedding_dim, pretrained_embeddings, 
                            model_save_path, max_len)
    elif model_type == 'bert':
        model = train_bert(train_df, val_df, test_df, model_save_path, max_len, early_stopping_patience)
    else:
        raise ValueError('Invalid model_type')

    # Save final model to specified path
    model_save_file = os.path.join(model_save_path, f'spam_{model_type}_final.pt')
    model.save(model_save_file)
    print(f"Final model saved to {model_save_file}")

    if evaluate:
        print("\nEvaluating on test set:")
        # Build vocabulary from training data only for non-BERT models
        if model_type != 'bert':
            word2idx, idx2word = build_vocab(train_df['text'])
            X_test = torch.tensor([encode(t, word2idx, max_len) for t in test_df['text']])
            y_test = torch.tensor(test_df['label'].values, dtype=torch.float32)
            evaluate_model(model, model_type, X_test=X_test, y_test=y_test)
        else:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            test_encodings = tokenizer(test_df['text'].tolist(), padding='max_length', 
                                     truncation=True, max_length=max_len, return_tensors='pt')
            y_test = torch.tensor(test_df['label'].values, dtype=torch.float32)
            test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], y_test)
            test_loader = DataLoader(test_dataset, batch_size=32)
            evaluate_model(model, model_type, test_loader=test_loader)

    return model


def evaluate_model(model, model_type, X_test=None, y_test=None, test_loader=None):
    """
    Evaluate model performance on test data
    Args:
        model: Trained model instance
        model_type: Type of model ('cnn', 'bilstm', or 'bert')
        X_test: Test inputs (for CNN and BiLSTM)
        y_test: Test labels
        test_loader: Test data loader (for BERT)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    if model_type == 'bert':
        y_pred, y_true = [], []
        for batch in test_loader:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                outputs = model(**inputs)

            # Fix: Handle tuple output from SpamBERT model
            if isinstance(outputs, tuple):
                probs = outputs[0]  # First element contains the probabilities
            else:
                probs = outputs

            predictions = (probs > 0.5).long().cpu().numpy()
            y_pred.extend(predictions)
            y_true.extend(batch[2].cpu().numpy())

        print(classification_report(y_true, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    elif model_type in ['cnn', 'bilstm']:
        X_test = X_test.to(device)
        with torch.no_grad():
            outputs = model(X_test)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            # Check tensor dimensions before squeezing
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            predictions = (outputs > 0.5).cpu().numpy()

        print(classification_report(y_test.numpy(), predictions))
        print("Confusion Matrix:\n", confusion_matrix(y_test.numpy(), predictions))
