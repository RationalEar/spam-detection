import torch
from torch.utils.data import Dataset


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts.reset_index(drop=True)  # Ensure continuous indices
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        try:
            text = str(self.texts.iloc[idx])  # Use iloc for positional indexing
            label = int(self.labels.iloc[idx])
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error processing index {idx}: {str(e)}")
            # Return empty sample (handle this in your training loop)
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'label': torch.tensor(-1, dtype=torch.long)  # Special invalid label
            }
