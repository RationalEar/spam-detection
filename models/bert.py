from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import BertModel


class SpamBERT(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=1, dropout=0.1):
        super(SpamBERT, self).__init__()
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        # Enable gradient checkpointing for memory efficiency
        self.bert.gradient_checkpointing_enable()

        # Initialize explainability components
        self._init_explainability()

        # Temperature parameter for attention normalization
        self.attention_temperature = 0.1

    def _init_explainability(self):
        """Initialize explainability components"""
        self.layer_integrated_gradients = None  # Lazy initialization
        self.target_layers = [6, 9, 12]  # Key layers for attention analysis

    def forward_for_ig(self, inputs, attention_mask=None, token_type_ids=None):
        """Forward pass specifically for integrated gradients"""
        if isinstance(inputs, tuple):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else attention_mask
            token_type_ids = inputs[2] if len(inputs) > 2 else token_type_ids
        else:
            input_ids = inputs

        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]

    @autocast('cuda')  # Enable automatic mixed precision
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                return_attentions=False) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with optional attention analysis
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            token_type_ids: Token type ids
            return_attentions: Whether to return attention maps
        Returns:
            tuple: (probabilities, attention_data)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=return_attentions
        )

        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        probs = torch.sigmoid(logits).squeeze(-1)

        attention_data = None
        if return_attentions:
            # Get attention weights from all layers
            if hasattr(outputs, 'attentions') and outputs.attentions:
                attention_data = self._process_attention_weights(outputs.attentions)
            else:
                print("Warning: No attention weights available from BERT outputs")

        return probs, attention_data

    def _process_attention_weights(self, attention_weights: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
        """
        Process attention weights with temperature scaling
        Args:
            attention_weights: Tuple of attention tensors
        Returns:
            dict: Processed attention weights for target layers
        """
        processed_attentions = {}
        for layer_idx in self.target_layers:
            if layer_idx <= len(attention_weights):
                # Get attention weights for this layer (average over attention heads)
                layer_attention = attention_weights[layer_idx - 1].mean(dim=1)  # Average over heads
                # Apply temperature scaling
                scaled_attention = layer_attention / self.attention_temperature
                processed_attentions[f'layer_{layer_idx}'] = torch.softmax(scaled_attention, dim=-1).detach()
        return processed_attentions

    def get_explanation_metrics(self, attributions: torch.Tensor, delta: torch.Tensor) -> Dict[str, float]:
        """
        Calculate explanation quality metrics
        Args:
            attributions: Attribution scores
            delta: Convergence delta
        Returns:
            dict: Metrics including faithfulness and stability scores
        """
        metrics = {
            'convergence_delta': float(delta.mean().item()),
            'attribution_mean': float(attributions.mean().item()),
            'attribution_std': float(attributions.std().item())
        }
        return metrics

    def save(self, path: str):
        """Save model state"""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load model state and set to eval mode"""
        self.load_state_dict(torch.load(path))
        self.eval()
