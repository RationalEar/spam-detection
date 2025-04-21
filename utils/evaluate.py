import torch


def evaluate(model, dataloader, device):
    """Calculate accuracy on validation/test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


def check_data_integrity(df):
    print(f"Total samples: {len(df)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Index range: {df.index.min()} to {df.index.max()}")
    print(f"Index continuity check: {(df.index == range(len(df))).all()}")
