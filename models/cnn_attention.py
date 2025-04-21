import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        attn_weights = self.attention(x)  # (batch_size, seq_len, 1)
        return attn_weights.squeeze(-1)  # (batch_size, seq_len)


class SpamCNNAttention(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=300):
        super().__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        
        # Attention
        self.attention = AttentionLayer(64)
        
        # Classifier
        self.fc = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # (batch, seq, embed)
        x = x.permute(0, 2, 1)  # (batch, embed, seq)
        
        # Convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # (batch, 64, seq)
        x = x.permute(0, 2, 1)  # (batch, seq, 64)
        
        # Attention
        attn_weights = self.attention(x)  # (batch, seq)
        x = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (batch, 64)
        
        # Classification
        x = self.dropout(x)
        return self.fc(x)
    
    def visualize_attention(self, text, tokenizer):
        self.eval()
        tokens = tokenizer(text, return_tensors='pt')['input_ids'].to(self.device)
        
        with torch.no_grad():
            x = self.embedding(tokens).permute(0, 2, 1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x)).permute(0, 2, 1)
            attn_weights = self.attention(x)
        
        return {
            'tokens': tokenizer.convert_ids_to_tokens(tokens[0]),
            'attention': attn_weights[0].cpu().numpy()
        }
