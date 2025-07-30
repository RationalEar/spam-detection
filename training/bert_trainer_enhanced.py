import os

import mlflow
import torch
from mlflow.entities import SpanType
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

from models.bert import SpamBERT


class LabelSmoothingBCELoss(nn.Module):
    """Label smoothing for binary classification"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCELoss()
    
    def forward(self, pred, target):
        # Apply label smoothing
        smoothed_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(pred, smoothed_target)


@mlflow.trace(name = "Model Training", span_type=SpanType.CHAIN)
def get_layerwise_optimizer(model, base_lr=2e-5, decay_factor=0.9):
    """
    Create optimizer with layer-wise learning rate decay
    Lower layers get lower learning rates
    """
    parameters = []
    
    # BERT embeddings (lowest layer)
    parameters.append({
        'params': model.bert.embeddings.parameters(),
        'lr': base_lr * (decay_factor ** 12)
    })
    
    # BERT encoder layers (12 layers)
    for i, layer in enumerate(model.bert.encoder.layer):
        lr = base_lr * (decay_factor ** (11 - i))  # Higher layers get higher LR
        parameters.append({
            'params': layer.parameters(),
            'lr': lr
        })
    
    # BERT pooler
    parameters.append({
        'params': model.bert.pooler.parameters(),
        'lr': base_lr
    })
    
    # Classification head (highest learning rate)
    parameters.append({
        'params': [
            *model.dropout.parameters(),
            *model.classifier.parameters()
        ],
        'lr': base_lr * 2  # Classification head gets 2x base LR
    })
    
    return optim.AdamW(parameters, weight_decay=0.01)


@mlflow.trace(name = "Model Training", span_type=SpanType.CHAIN)
def train_bert(train_df, val_df, test_df, model_save_path='', max_len=200, early_stopping_patience=5):
    """
    Train BERT model with enhanced regularization and optimization
    
    Args:
        train_df: DataFrame containing training data
        val_df: DataFrame containing validation data
        test_df: DataFrame containing test data
        model_save_path: Directory to save model checkpoints
        max_len: Maximum sequence length
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped
    """
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def bert_encode(texts, tokenizer, max_len=200):
        """Tokenize and encode sequences for BERT"""
        return tokenizer(texts.tolist(), padding='max_length', truncation=True, 
                         max_length=max_len, return_tensors='pt')
    
    # Encode all datasets
    train_encodings = bert_encode(train_df['text'], tokenizer, max_len)
    val_encodings = bert_encode(val_df['text'], tokenizer, max_len)
    
    y_train = torch.tensor(train_df['label'].values, dtype=torch.float32)
    y_val = torch.tensor(val_df['label'].values, dtype=torch.float32)
    
    # Initialize BERT model with dropout=0.2
    model = SpamBERT(dropout=0.2)
    
    # Training parameters
    epochs = 10
    learning_rate = 2e-5
    batch_size = 32
    max_grad_norm = 1.0  # Gradient clipping
    label_smoothing = 0.1
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function with label smoothing and optimizer with layer-wise decay
    criterion = LabelSmoothingBCELoss(smoothing=label_smoothing)
    optimizer = get_layerwise_optimizer(model, base_lr=learning_rate)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Training BERT with enhanced regularization:")
    print(f"- Dropout: 0.2")
    print(f"- Label smoothing: {label_smoothing}")
    print(f"- Gradient clipping: {max_grad_norm}")
    print(f"- Layer-wise learning rate decay")
    print(f"- Learning rate: {learning_rate}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Check tensor dimensions before squeezing
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            
            # Ensure consistent dtypes for loss calculation
            outputs = outputs.float()
            labels = labels.float()
                
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
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
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Check tensor dimensions before squeezing
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                
                # Ensure consistent dtypes for validation loss
                outputs = outputs.float()
                labels = labels.float()
                
                # Use standard BCE for validation (no label smoothing)
                val_loss = nn.BCELoss()(outputs, labels)
                total_val_loss += val_loss.item()
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
                best_model_path = os.path.join(model_save_path, 'best_bert_model.pt')
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
