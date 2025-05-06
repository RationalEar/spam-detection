import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

try:
    DIR_ROOT
except NameError:
    DIR_ROOT = '/content/drive/MyDrive/Colab Notebooks/spam-detection/'

def plot_metrics(train_losses, val_accuracies):
    if not os.path.exists(DIR_ROOT + 'images'):
        os.makedirs(DIR_ROOT + 'images')
    
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.savefig(DIR_ROOT + 'images/training_metrics.png')
    plt.show()


def plot_confusion_matrix(model, dataloader):
    if not os.path.exists(DIR_ROOT + 'images'):
        os.makedirs(DIR_ROOT + 'images')
    model.eval()
    all_preds = []
    all_labels = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(DIR_ROOT + 'images/confusion_matrix.png')
    plt.show()


# Add attention visualization
def visualize_sample(model, tokenizer, text="Claim your free prize now!"):
    model.eval()
    result = model.visualize_attention(text, tokenizer)
    
    plt.figure(figsize=(10, 2))
    plt.bar(range(len(result['tokens'])), result['attention'],
            width=0.8, color='skyblue')
    plt.xticks(range(len(result['tokens'])), result['tokens'], rotation=90)
    plt.title('Attention Weights')
    plt.show()


def compare_models(cnn_metrics, attn_metrics, save_path='results/comparison.png'):
    """
    Compare training metrics between CNN and CNN+Attention models

    Args:
        cnn_metrics: Path to JSON or dict of CNN metrics {'loss': [], 'acc': []}
        attn_metrics: Path to JSON or dict of Attention metrics {'loss': [], 'acc': []}
        save_path: Where to save the comparison plot
    """
    # Load metrics if paths are provided
    if isinstance(cnn_metrics, (str, Path)):
        with open(cnn_metrics) as f:
            cnn_metrics = json.load(f)
    if isinstance(attn_metrics, (str, Path)):
        with open(attn_metrics) as f:
            attn_metrics = json.load(f)
    
    # Create figure
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(cnn_metrics['train_losses'], label='CNN', color='blue')
    plt.plot(attn_metrics['train_losses'], label='CNN+Attention', color='orange')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(cnn_metrics['val_accuracies'], label='CNN', color='blue')
    plt.plot(attn_metrics['val_accuracies'], label='CNN+Attention', color='orange')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Save and show
    Path(DIR_ROOT + save_path).parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(DIR_ROOT + save_path)
    plt.show()
    
    # Print final metrics comparison
    print("\n=== Final Metrics ===")
    print(f"{'Model':<15} | {'Best Loss':<10} | {'Best Accuracy':<10}")
    print("-" * 40)
    print(f"{'CNN':<15} | {min(cnn_metrics['train_losses']):<10.4f} | {max(cnn_metrics['val_accuracies']):<10.2f}%")
    print(
        f"{'CNN+Attention':<15} | {min(attn_metrics['train_losses']):<10.4f} | {max(attn_metrics['val_accuracies']):<10.2f}%")

