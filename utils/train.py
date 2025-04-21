import os

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from models.cnn_attention import SpamCNNAttention
from utils.EmailDataset import EmailDataset
from utils.evaluate import evaluate


def train_model(train_df, test_df, batch_size=32, epochs=10, learning_rate=1e-3):
    """Complete training workflow with metrics tracking"""
    # Initialize
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = EmailDataset(train_df['text'], train_df['label'], tokenizer)
    test_dataset = EmailDataset(test_df['text'], test_df['label'], tokenizer)
    
    # DataLoaders
    num_workers = min(os.cpu_count(), len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Model (using CNN with Attention)
    model = SpamCNNAttention(vocab_size=tokenizer.vocab_size).to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Metrics tracking
    train_losses = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        val_acc = evaluate(model, test_loader, device)
        
        # Store metrics
        train_losses.append(epoch_loss / len(train_loader))
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch + 1}: Loss={train_losses[-1]:.4f}, Acc={val_acc:.2f}%')
    
    return model, train_losses, val_accuracies
