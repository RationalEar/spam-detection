import torch.nn as nn
import torch.nn.functional as F


class SpamCNN(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=300, num_classes=2):
        super(SpamCNN, self).__init__()
        
        # Embedding layer (pretrained GloVe recommended)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=7, padding=3)
        
        # Global max pooling
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Classifier
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, seq_length)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Pooling
        x = self.pool(x).squeeze(-1)  # (batch_size, num_filters)
        
        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class SpamCNNWithCAM(SpamCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradients = None
        self.activations = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        
        # Register hook for last conv layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        h = x.register_hook(self.activations_hook)
        self.activations = x
        
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.activations
