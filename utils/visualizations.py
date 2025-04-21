import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

def plot_metrics(train_losses, val_accuracies):
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
    
    plt.savefig('images/training_metrics.png')
    plt.show()


def plot_confusion_matrix(model, dataloader):
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
    plt.savefig('images/confusion_matrix.png')
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