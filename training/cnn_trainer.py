import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from models.cnn import SpamCNN
from utils.functions import build_vocab, encode


def train_cnn(train_df, val_df, test_df, embedding_dim=300, pretrained_embeddings=None, 
             model_save_path='', max_len=200, early_stopping_patience=5):
    """
    Train CNN model with early stopping
    
    Args:
        train_df: DataFrame containing training data
        val_df: DataFrame containing validation data
        test_df: DataFrame containing test data
        embedding_dim: Dimension of word embeddings
        pretrained_embeddings: Pre-trained embeddings tensor
        model_save_path: Directory to save model checkpoints
        max_len: Maximum sequence length
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped
    """
    # Build vocabulary from training data only
    word2idx, idx2word = build_vocab(train_df['text'])

    # Encode all datasets
    X_train = torch.tensor([encode(t, word2idx, max_len) for t in train_df['text']])
    y_train = torch.tensor(train_df['label'].values, dtype=torch.float32)
    
    X_val = torch.tensor([encode(t, word2idx, max_len) for t in val_df['text']])
    y_val = torch.tensor(val_df['label'].values, dtype=torch.float32)
    
    # Initialize model
    model = SpamCNN(vocab_size=len(word2idx), embedding_dim=embedding_dim,
                    pretrained_embeddings=pretrained_embeddings)
    
    # Training parameters
    epochs = 50
    learning_rate = 1e-3
    batch_size = 32
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        for batch in train_loader:
            inputs, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Check tensor dimensions before squeezing
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = total_train_loss / train_steps
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = [b.to(device) for b in batch]
                outputs = model(inputs)
                
                # Check tensor dimensions before squeezing
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = total_val_loss / val_steps
        
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save best model checkpoint
            if model_save_path:
                best_model_path = os.path.join(model_save_path, 'best_cnn_model.pt')
                torch.save(best_model_state, best_model_path)
                print(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model for return
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model
