from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients
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
        
    def _get_layer_integrated_gradients(self):
        """Lazy initialization of LayerIntegratedGradients"""
        if self.layer_integrated_gradients is None:
            self.layer_integrated_gradients = LayerIntegratedGradients(
                self.forward_for_ig,
                self.bert.embeddings
            )
        return self.layer_integrated_gradients
    
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
    
    def compute_integrated_gradients(self, input_ids, attention_mask=None, token_type_ids=None, 
                                   n_steps=50, chunk_size=128) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute integrated gradients for input attribution with memory optimization
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            token_type_ids: Token type ids
            n_steps: Number of steps for approximation
            chunk_size: Size of chunks to process at once
        Returns:
            tuple: (attributions, delta)
        """
        lig = self._get_layer_integrated_gradients()
        device = next(self.parameters()).device
        
        # Move inputs to CPU initially
        input_ids = input_ids.cpu()
        if attention_mask is not None:
            attention_mask = attention_mask.cpu()
        if token_type_ids is not None:
            token_type_ids = token_type_ids.cpu()
            
        # Process in chunks
        total_len = input_ids.size(1)
        all_attributions = []
        all_deltas = []
        
        for start_idx in range(0, total_len, chunk_size):
            end_idx = min(start_idx + chunk_size, total_len)
            
            # Extract chunk
            chunk_input_ids = input_ids[:, start_idx:end_idx].to(device)
            chunk_attention = attention_mask[:, start_idx:end_idx].to(device) if attention_mask is not None else None
            chunk_token_types = token_type_ids[:, start_idx:end_idx].to(device) if token_type_ids is not None else None
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Compute attributions for chunk using FP16
            with autocast(device_type='cuda', dtype=torch.float16):
                chunk_attributions, chunk_delta = lig.attribute(
                    inputs=chunk_input_ids,
                    additional_forward_args=(chunk_attention, chunk_token_types),
                    n_steps=n_steps,
                    return_convergence_delta=True
                )
            
            # Move results back to CPU
            all_attributions.append(chunk_attributions.cpu())
            all_deltas.append(chunk_delta.cpu())
            
            # Clear CUDA cache again
            torch.cuda.empty_cache()
        
        # Combine results
        attributions = torch.cat(all_attributions, dim=1)
        delta = torch.stack(all_deltas).mean()
        
        return attributions.to(device), delta.to(device)
    
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
