import json
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from models.cnn import SpamCNN
from models.cnn_attention import SpamCNNAttention
from utils.data_loader import get_train_loader, get_test_loader
from utils.evaluate import evaluate
from utils.visualizations import compare_models

try:
    DIR_ROOT
except NameError:
    DIR_ROOT = '/content/drive/MyDrive/Colab Notebooks/spam-detection/'


def train_single_model(model_class, model_name, train_loader, test_loader, epochs=10, learning_rate=1e-3):
    """Train and evaluate one model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Tracking
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Training
        for batch in tqdm(train_loader, desc=f'{model_name} Epoch {epoch + 1}'):
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Evaluation
        val_acc = evaluate(model, test_loader, device)
        train_losses.append(epoch_loss / len(train_loader))
        val_accuracies.append(val_acc)
        
        print(f'{model_name} Epoch {epoch + 1}: Loss={train_losses[-1]:.4f}, Acc={val_acc:.2f}%')
    
    # Save model and metrics
    os.makedirs(DIR_ROOT + 'saved_models', exist_ok=True)
    torch.save(model.state_dict(), f'{DIR_ROOT}saved_models/{model_name}.pt')
    with open(f'{DIR_ROOT}saved_models/{model_name}_metrics.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'val_accuracies': val_accuracies}, f)
    
    return model, train_losses, val_accuracies


def train_all_models():
    # Load data
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    
    # Train both models
    print("=== Training SpamCNN ===")
    cnn_model, cnn_losses, cnn_accs = train_single_model(
        SpamCNN, 'spam_cnn', train_loader, test_loader)
    
    print("\n=== Training SpamCNNAttention ===")
    attn_model, attn_losses, attn_accs = train_single_model(
        SpamCNNAttention, 'spam_cnn_attention', train_loader, test_loader)
    
    # Comparative analysis
    compare_models(
        cnn_metrics={'loss': cnn_losses, 'acc': cnn_accs},
        attn_metrics={'loss': attn_losses, 'acc': attn_accs}
    )
    
    return {
        'cnn': cnn_model,
        'cnn_attention': attn_model
    }

# if __name__ == '__main__':
#     trained_models = train_all_models()
