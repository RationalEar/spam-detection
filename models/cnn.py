import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.cnn_metrics import (
    compute_metrics, 
    compute_explanation_metrics,
    generate_adversarial_example,
    measure_adversarial_robustness,
    evaluate_adversarial_examples
)
from integrations.grad_cam import grad_cam, grad_cam_auto, grad_cam_with_timing


class SpamCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings, num_classes=1, dropout=0.5):
        super(SpamCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(pretrained_embeddings, requires_grad=False)

        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=7, padding=3)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_max_pool(x).squeeze(-1)  # (batch_size, 32)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x  # shape: (batch_size, 1)
    
    def grad_cam(self, x, target_class=None):
        """
        Wrapper for grad_cam function from integrations.grad_cam
        """
        return grad_cam_with_timing(self, x, target_class)
    
    def grad_cam_auto(self, x, target_class=None):
        """
        Wrapper for grad_cam_auto function from integrations.grad_cam
        """
        return grad_cam_auto(self, x, target_class)

    def predict(self, x):
        """
        Make predictions on input data
        Args:
            x: input tensor (batch_size, seq_len)
        Returns:
            predictions: tensor of predictions (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs.squeeze(1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        if map_location is None:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.eval()
        
    def compute_explanation_metrics(self, x, cam_maps, num_perturbations=10):
        return compute_explanation_metrics(self, x, cam_maps, num_perturbations)

    def generate_adversarial_example(self, x, y, epsilon=0.1, num_steps=10):
        return generate_adversarial_example(self, x, y, epsilon, num_steps)

    def measure_adversarial_robustness(self, x, y, epsilon_range=(0.01, 0.05, 0.1)):
        return measure_adversarial_robustness(self, x, y, epsilon_range)

    def evaluate_adversarial_examples(self, x, y):
        return evaluate_adversarial_examples(self, x, y)
