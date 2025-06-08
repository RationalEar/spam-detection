import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output, mask=None):
        # lstm_output: (batch_size, seq_len, hidden_dim*2)
        attn_weights = self.attn(lstm_output).squeeze(-1)  # (batch_size, seq_len)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch_size, hidden_dim*2)
        return context, attn_weights


class BiLSTMSpam(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings, hidden_dim=128, num_layers=2, num_classes=1, dropout=0.5):
        super(BiLSTMSpam, self).__init__()
        self.config = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'dropout': dropout
        }
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(pretrained_embeddings, requires_grad=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len)
        if isinstance(x, torch.LongTensor) or (torch.cuda.is_available() and isinstance(x, torch.cuda.LongTensor)):
            # Regular forward pass with input indices
            x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        # If x is already an embedding tensor, use it directly
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        context, attn_weights = self.attention(lstm_out, mask)
        x = self.dropout(F.relu(self.fc1(context)))
        x = self.fc2(x)
        x = torch.sigmoid(x).squeeze(-1)
        return x, attn_weights

    def generate_adversarial_example(self, x, y, epsilon=0.1, num_steps=30):
        """
        Generate adversarial examples using Projected Gradient Descent (PGD)
        Args:
            x: Input tensor (batch_size, seq_len)
            y: Target labels
            epsilon: Maximum perturbation size
            num_steps: Number of optimization steps (increased to 30)
        Returns:
            Adversarial examples as token indices
        """
        self.train()  # Enable gradients
        
        # Ensure inputs are on the correct device
        device = next(self.parameters()).device
        x = x.to(device)
        y = y.to(device)
        
        # Get initial embeddings
        embeddings = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Create adversarial embeddings starting from original embeddings
        emb_adv = embeddings.clone().detach().requires_grad_(True)
        criterion = nn.BCELoss()
        
        # Target is the opposite of the true label
        target = 1 - y
        
        print("\nStarting adversarial example generation:")
        print("Initial target labels:", target.cpu().numpy())
        
        # Get initial predictions
        with torch.no_grad():
            initial_outputs, _ = self(x)
            print("Initial predictions:", initial_outputs.cpu().numpy())
        
        # PGD attack
        alpha = epsilon / num_steps  # Step size
        best_loss = float('-inf')
        best_emb_adv = None
        
        for step in range(num_steps):
            # Forward pass through LSTM layers with current adversarial embeddings
            batch_size = emb_adv.size(0)
            
            # Pack the embeddings to handle variable lengths
            lengths = (x != 0).sum(dim=1)  # Get actual sequence lengths
            packed_embeddings = pack_padded_sequence(emb_adv, lengths.cpu(), batch_first=True, enforce_sorted=False)
            
            # Pass through LSTM
            lstm_out, _ = self.lstm(packed_embeddings)
            
            # Unpack the sequence
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            
            # Apply attention
            if hasattr(self, 'attention'):
                attn_mask = (x != 0).float()  # Create attention mask
                context, attn_weights = self.attention(lstm_out, attn_mask)
            else:
                # If no attention, use last hidden state
                context = lstm_out[:, -1, :]
            
            # Final classification
            x_fc1 = self.dropout(F.relu(self.fc1(context)))
            outputs = torch.sigmoid(self.fc2(x_fc1))
            
            # Print intermediate predictions every few steps
            if step % 5 == 0:
                print(f"\nStep {step} predictions:", outputs.detach().cpu().numpy())
            
            # Compute loss to maximize probability of target (opposite) class
            loss = -criterion(outputs.squeeze(), target)  # Negative loss to maximize target class probability
            
            # Track best adversarial example
            if loss.item() > best_loss:
                best_loss = loss.item()
                best_emb_adv = emb_adv.clone().detach()
            
            # Print loss
            if step % 5 == 0:
                print(f"Step {step} loss:", loss.item())
            
            # Compute gradients
            loss.backward()
            
            # Update adversarial embeddings with normalized gradients
            with torch.no_grad():
                # Normalize gradients
                grad_norm = torch.norm(emb_adv.grad, dim=2, keepdim=True)
                normalized_grad = emb_adv.grad / (grad_norm + 1e-8)
                
                # Update embeddings with normalized gradients
                emb_adv.data = emb_adv.data - alpha * normalized_grad
                
                # Project back to epsilon ball around original embeddings
                delta = emb_adv.data - embeddings
                norm = torch.norm(delta, dim=2, keepdim=True)
                factor = torch.min(norm, torch.ones_like(norm) * epsilon) / (norm + 1e-8)
                delta = delta * factor
                emb_adv.data = embeddings + delta
                
                # Reset gradients for next step
                if step < num_steps - 1:
                    emb_adv.grad.zero_()
        
        print("\nConverting perturbed embeddings to tokens...")
        
        # Use the best adversarial example found
        emb_adv = best_emb_adv if best_emb_adv is not None else emb_adv
        
        # Convert perturbed embeddings back to token indices
        with torch.no_grad():
            # Get embedding matrix
            emb_matrix = self.embedding.weight  # (vocab_size, embedding_dim)
            
            # Initialize output tensor
            batch_size, seq_len, emb_dim = emb_adv.size()
            x_adv = x.clone()  # Start with original tokens
            
            # Process embeddings in smaller batches to avoid memory issues
            batch_size_inner = 128  # Process 128 tokens at a time
            emb_adv_flat = emb_adv.view(-1, emb_dim)
            
            # Track original tokens for each position
            original_tokens = x.view(-1)
            
            # Track number of changes per sequence
            changes_per_seq = torch.zeros(batch_size, dtype=torch.long)
            max_changes_per_seq = min(30, seq_len // 2)  # Increased from 20 to 30, but limit to half sequence length
            
            # Calculate importance scores for each position using attention weights
            importance_scores = torch.zeros(batch_size * seq_len, device=device)
            
            # Use attention weights if available
            if hasattr(self, 'attention'):
                # Reshape attention weights to match token positions
                attn_weights = attn_weights.view(-1)
                # Scale attention weights by prediction confidence and target difference
                pred_conf = outputs.detach().squeeze()
                pred_diff = torch.abs(pred_conf - target)  # Higher weight for tokens that could flip prediction
                pred_conf = torch.where(target == 1, pred_conf, 1 - pred_conf)
                pred_conf = pred_conf.repeat_interleave(seq_len)
                importance_scores = attn_weights * pred_conf * pred_diff.repeat_interleave(seq_len)
            else:
                # Fallback to L2 distance if no attention
                for i in range(0, emb_adv_flat.size(0), batch_size_inner):
                    start_idx = i
                    end_idx = min(i + batch_size_inner, emb_adv_flat.size(0))
                    orig_emb = self.embedding(original_tokens[start_idx:end_idx])
                    perturbed_emb = emb_adv_flat[start_idx:end_idx]
                    importance_scores[start_idx:end_idx] = torch.norm(perturbed_emb - orig_emb, dim=1)
            
            # Sort positions by importance score
            sorted_positions = torch.argsort(importance_scores, descending=True)
            
            # More aggressive similarity threshold that scales with epsilon and confidence
            base_threshold = 0.3  # Lower base threshold
            conf_factor = pred_conf.mean().item()  # Use prediction confidence to adjust threshold
            similarity_threshold = max(0.2, base_threshold - epsilon * 2 - (1 - conf_factor) * 0.1)
            
            # Try to modify most important positions first
            for pos in sorted_positions:
                seq_idx = pos.item() // seq_len
                if changes_per_seq[seq_idx] >= max_changes_per_seq:
                    continue
                
                orig_token = original_tokens[pos].item()
                if orig_token == 0:  # Skip PAD tokens
                    continue
                
                # Get current embedding
                current_emb = emb_adv_flat[pos].unsqueeze(0)  # (1, emb_dim)
                
                # Compute cosine similarity with all tokens
                current_emb_norm = F.normalize(current_emb, p=2, dim=1)
                emb_matrix_norm = F.normalize(emb_matrix, p=2, dim=1)
                similarities = torch.mm(current_emb_norm, emb_matrix_norm.t()).squeeze()
                
                # Get top-k similar tokens (increased from 30 to 40)
                k = 40
                top_k_similarities, top_k_indices = torch.topk(similarities, k=k)
                
                # Try each candidate and pick the one that moves prediction most towards target
                best_token = orig_token
                max_impact = -1.0
                
                # Create a mini-batch with each candidate
                test_sequences = x_adv.clone().view(-1)
                
                # Try candidates in batches to speed up evaluation
                batch_size_cand = 10
                for i in range(0, k, batch_size_cand):
                    end_idx = min(i + batch_size_cand, k)
                    curr_candidates = top_k_indices[i:end_idx]
                    curr_similarities = top_k_similarities[i:end_idx]
                    
                    # Skip candidates that don't meet similarity threshold
                    valid_mask = curr_similarities >= similarity_threshold
                    if not valid_mask.any():
                        continue
                    
                    curr_candidates = curr_candidates[valid_mask]
                    
                    # Create batch of sequences with different candidates
                    test_batch = []
                    for token_idx in curr_candidates:
                        if token_idx.item() == orig_token:
                            continue
                        temp_seq = test_sequences.clone()
                        temp_seq[pos] = token_idx
                        test_batch.append(temp_seq.view(batch_size, seq_len))
                    
                    if not test_batch:
                        continue
                    
                    # Evaluate all candidates at once
                    test_batch = torch.stack(test_batch)  # Shape: (num_candidates, batch_size, seq_len)
                    # Reshape to (num_candidates * batch_size, seq_len)
                    test_batch_flat = test_batch.view(-1, seq_len)
                    outputs_batch, _ = self(test_batch_flat)
                    # Reshape outputs back to (num_candidates, batch_size)
                    outputs_batch = outputs_batch.view(-1, batch_size)
                    
                    # Calculate impact for each candidate
                    target_expanded = target.unsqueeze(0).expand(len(test_batch), -1)
                    impact_batch = torch.abs(outputs_batch - y)
                    
                    # Find best candidate
                    max_impact_idx = impact_batch.mean(dim=1).argmax()
                    curr_impact = impact_batch.mean(dim=1)[max_impact_idx].item()
                    
                    if curr_impact > max_impact:
                        max_impact = curr_impact
                        # Get the token from the original shaped test_batch
                        seq_idx = pos // seq_len  # Get the sequence index
                        token_pos = pos % seq_len  # Get position within sequence
                        best_token = test_batch[max_impact_idx, seq_idx, token_pos].item()
                
                # Apply the best change if we found one
                if best_token != orig_token:
                    x_adv.view(-1)[pos] = best_token
                    changes_per_seq[seq_idx] += 1
            
            # Get final predictions
            final_outputs, _ = self(x_adv)
            print("\nFinal adversarial predictions:", final_outputs.cpu().numpy())
            print("Number of tokens changed:", (x_adv != x).sum().item())
            print("Changes per sequence:", changes_per_seq.tolist())
        
        self.eval()  # Reset to evaluation mode
        return x_adv

    def clip_gradients(self, max_norm=1.0):
        """
        Clips gradients of the model parameters.
        Args:
            max_norm: Maximum norm for gradient clipping
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    def update_training_history(self, train_loss, val_loss, train_acc, val_acc):
        """
        Updates the training history with new metrics
        """
        self.training_history['train_loss'].append(float(train_loss))
        self.training_history['val_loss'].append(float(val_loss))
        self.training_history['train_acc'].append(float(train_acc))
        self.training_history['val_acc'].append(float(val_acc))

    def save(self, path):
        """
        Save model state, configuration, and training history
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, path)
        
        # Save config and history separately for easy access
        config_path = os.path.join(os.path.dirname(path), 'config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'config': self.config,
                'training_history': self.training_history
            }, f, indent=4)

    def load(self, path):
        """
        Load model state, configuration, and training history
        """
        checkpoint = torch.load(path)
        
        # Load configuration if available
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        
        # Load training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)
        
        self.eval()
