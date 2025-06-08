import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import spearmanr # For Rank Correlation

from utils.cnn_evaluation import compute_metrics


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
        Compute Grad-CAM for the input batch x.
        Args:
            x: input tensor (batch_size, seq_len)
            target_class: index of the class to compute Grad-CAM for (default: predicted class)
        Returns:
            cam: class activation map (batch_size, seq_len)
        """
        self.eval()
        # Force CPU computation to avoid CUDA errors
        self.to('cpu')

        # Ensure x is on CPU and remains a LongTensor for embedding
        x_cpu = x.detach().cpu()
        if target_class is not None:
            target_class = target_class.detach().cpu()

        # Set up for gradient capture
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output.detach())

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        # Register hooks
        handle_fwd = self.conv3.register_forward_hook(forward_hook)
        handle_bwd = self.conv3.register_full_backward_hook(backward_hook)

        try:
            # Forward pass
            with torch.set_grad_enabled(True):
                # Pass through model
                x_emb = self.embedding(x_cpu)  # x must remain a LongTensor
                x_perm = x_emb.permute(0, 2, 1)
                x1 = F.relu(self.conv1(x_perm))
                x2 = F.relu(self.conv2(x1))
                x3 = F.relu(self.conv3(x2))
                pooled = self.global_max_pool(x3).squeeze(-1)
                x_fc1 = self.dropout(F.relu(self.fc1(pooled)))
                logits = self.fc2(x_fc1)

                # For binary classification
                if target_class is None:
                    # Just use logits for gradient
                    loss = logits.sum()
                else:
                    # Target-specific loss
                    loss = (logits * target_class.float()).sum()

                # Compute gradients
                self.zero_grad()
                loss.backward(retain_graph=True)

            # Ensure we have activations and gradients
            if not activations or not gradients:
                raise ValueError("No activations or gradients captured")

            # Get activation maps and gradients
            act = activations[0]  # (batch, channels, seq_len)
            grad = gradients[0]  # (batch, channels, seq_len)

            # Compute importance weights
            weights = grad.mean(dim=2, keepdim=True)  # (batch, channels, 1)

            # Compute weighted activations
            cam = (weights * act).sum(dim=1)  # (batch, seq_len)
            cam = F.relu(cam)  # Apply ReLU to focus on positive contributions

            # Normalize each CAM individually
            batch_size = cam.size(0)
            for i in range(batch_size):
                cam_min = cam[i].min()
                cam_max = cam[i].max()
                if cam_max > cam_min:  # Avoid division by zero
                    cam[i] = (cam[i] - cam_min) / (cam_max - cam_min)

            return cam

        except Exception as e:
            print(f"Error in grad_cam: {str(e)}")
            # Return uniform importance as fallback
            return torch.ones(x_cpu.size(0), x_cpu.size(1), dtype=torch.float)

        finally:
            # Always remove hooks
            handle_fwd.remove()
            handle_bwd.remove()

    def grad_cam_auto(self, x, target_class=None):
        """
        CUDA-based Grad-CAM implementation.
        Computes class activation maps for the input batch x using the final convolutional layer.
        
        Args:
            x: input tensor (batch_size, seq_len)
            target_class: index of the class to compute Grad-CAM for (default: predicted class)
        Returns:
            cam: class activation map (batch_size, seq_len)
        """
        self.eval()  # Ensure model is in evaluation mode

        # Ensure model and input are on CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This implementation requires CUDA.")

        self.cuda()  # Move model to CUDA
        x = x.cuda()  # Move input to CUDA
        if target_class is not None:
            target_class = target_class.cuda()

        # Initialize lists to store activations and gradients
        activations = []
        gradients = []

        def save_activation(module, input, output):
            activations.append(output.detach())

        def save_gradient(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        # Register hooks
        handle_fwd = self.conv3.register_forward_hook(save_activation)
        handle_bwd = self.conv3.register_full_backward_hook(save_gradient)

        try:
            # Forward pass with gradient computation
            with torch.set_grad_enabled(True):
                # Ensure input is CUDA LongTensor
                if not isinstance(x, torch.cuda.LongTensor):
                    x = x.long().cuda()

                # Get embeddings
                x_emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
                x_emb = x_emb.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)

                # Convolutional layers
                x1 = F.relu(self.conv1(x_emb))
                x2 = F.relu(self.conv2(x1))
                x3 = F.relu(self.conv3(x2))

                # Global max pooling and final layers
                pooled = self.global_max_pool(x3).squeeze(-1)
                x_fc1 = self.dropout(F.relu(self.fc1(pooled)))
                logits = self.fc2(x_fc1)

                if target_class is None:
                    # Just use logits for gradient
                    loss = logits.sum()
                else:
                    # Target-specific loss
                    loss = (logits * target_class.float()).sum()

                # Backward pass
                self.zero_grad(set_to_none=True)
                loss.backward()

            # Ensure we have activations and gradients
            if not activations or not gradients:
                raise ValueError("No activations or gradients captured")

            # Get activation maps and gradients
            act = activations[0]  # (batch_size, channels, seq_len)
            grad = gradients[0]  # (batch_size, channels, seq_len)

            # Compute importance weights
            weights = grad.mean(dim=2, keepdim=True)  # (batch_size, channels, 1)

            # Compute weighted activations
            cam = (weights * act).sum(dim=1)  # (batch_size, seq_len)

            # Apply ReLU and normalize per sample
            cam = F.relu(cam)

            # Normalize each CAM individually using vectorized operations
            cam_min = cam.min(dim=1, keepdim=True)[0]
            cam_max = cam.max(dim=1, keepdim=True)[0]
            cam = torch.where(
                cam_max > cam_min,
                (cam - cam_min) / (cam_max - cam_min + 1e-7),
                torch.zeros_like(cam)
            )

            return cam

        except Exception as e:
            print(f"Error in grad_cam_auto: {str(e)}")
            # Return uniform importance as fallback
            return torch.ones(x.size(0), x.size(1), device='cuda')

        finally:
            # Clean up hooks
            handle_fwd.remove()
            handle_bwd.remove()
            # Clear any remaining gradients
            self.zero_grad(set_to_none=True)

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
        """
        Compute explainability metrics including faithfulness and stability.
        Args:
            x: Input tensor (batch_size, seq_len)
            cam_maps: Grad-CAM activation maps (batch_size, seq_len)
            num_perturbations: Number of perturbations for stability testing
        Returns:
            dict: Dictionary containing explainability metrics
        """
        self.eval()

        # Ensure model is on CUDA if available
        if torch.cuda.is_available():
            self.cuda()
            device = 'cuda'
        else:
            self.cpu()
            device = 'cpu'

        # Move input tensors to the same device as model
        x = x.to(device)
        cam_maps = cam_maps.to(device)

        # Ensure embedding layer is on the correct device
        self.embedding = self.embedding.to(device)

        batch_size = x.size(0)

        def compute_auc_del(x_single, cam_single):
            # Ensure input tensors are on correct device
            x_single = x_single.to(device)
            cam_single = cam_single.to(device)

            # Sort indices by importance
            _, indices = torch.sort(cam_single, descending=True)
            deletions = []

            # Base prediction with all tokens
            with torch.no_grad():
                # Ensure input is LongTensor
                x_input = x_single.unsqueeze(0).long()
                base_pred = self(x_input).item()
            deletions.append(base_pred)

            # Create mask tensor for token deletion (using PAD token)
            pad_token = 0  # Assuming 0 is the PAD token
            x_perturbed = x_single.clone()

            # Progressively delete most important tokens
            for i in range(1, len(indices)):
                # Create a copy with top-i tokens masked as PAD
                x_perturbed = x_single.clone()
                x_perturbed[indices[:i]] = pad_token

                with torch.no_grad():
                    # Ensure input is LongTensor
                    x_input = x_perturbed.unsqueeze(0).long()
                    pred = self(x_input)
                deletions.append(pred.item())

            # Calculate AUC
            auc_del = np.trapz(deletions) / len(indices)
            return auc_del

        def compute_auc_ins(x_single, cam_single):
            # Ensure input tensors are on correct device
            x_single = x_single.to(device)
            cam_single = cam_single.to(device)

            # Sort indices by importance
            _, indices = torch.sort(cam_single, descending=True)
            insertions = []

            # Start with all tokens masked (PAD)
            pad_token = 0  # Assuming 0 is the PAD token
            x_perturbed = torch.ones_like(x_single).long() * pad_token

            # Get baseline prediction with all tokens masked
            with torch.no_grad():
                x_input = x_perturbed.unsqueeze(0)
                base_pred = self(x_input).item()
            insertions.append(base_pred)

            # Progressively insert most important tokens
            for i in range(1, len(indices)):
                # Reveal the top-i tokens
                x_perturbed = torch.ones_like(x_single).long() * pad_token
                x_perturbed[indices[:i]] = x_single[indices[:i]]

                with torch.no_grad():
                    x_input = x_perturbed.unsqueeze(0)
                    pred = self(x_input)
                insertions.append(pred.item())

            # Calculate AUC
            auc_ins = np.trapz(insertions) / len(indices)
            return auc_ins

        def compute_comprehensiveness_single(x_single, cam_single, k=5):
            """Computes comprehensiveness for a single sample."""
            x_single = x_single.to(device)
            cam_single = cam_single.to(device)

            pad_token = 0 # Assuming 0 is the PAD token
            num_features = x_single.size(0)
            actual_k = min(k, num_features)
            if actual_k == 0: return 0.0

            with torch.no_grad():
                original_pred = self(x_single.unsqueeze(0).long()).item()

            _, top_k_indices = torch.topk(cam_single, actual_k)
            
            x_masked = x_single.clone()
            x_masked[top_k_indices] = pad_token

            with torch.no_grad():
                pred_after_removal = self(x_masked.unsqueeze(0).long()).item()
            
            comprehensiveness = original_pred - pred_after_removal
            return comprehensiveness

        def compute_rank_correlation_single(cam_original, cam_perturbed):
            """Computes Spearman's rank correlation for a single pair of CAMs."""
            cam_original_flat = cam_original.flatten().cpu().numpy()
            cam_perturbed_flat = cam_perturbed.flatten().cpu().numpy()

            valid_mask = np.isfinite(cam_original_flat) & np.isfinite(cam_perturbed_flat)
            cam_original_valid = cam_original_flat[valid_mask]
            cam_perturbed_valid = cam_perturbed_flat[valid_mask]

            if len(cam_original_valid) < 2 or len(cam_perturbed_valid) < 2:
                return np.nan # Not enough data points
            
            correlation, _ = spearmanr(cam_original_valid, cam_perturbed_valid)
            return correlation

        # Calculate metrics for each sample in batch
        metrics = {
            'auc_del': [],
            'auc_ins': [],
            'jaccard_stability': [], # Renamed from 'stability'
            'comprehensiveness': [],
            'rank_correlation': [],
            'ecs': []  # Explanation Consistency Score
        }

        k_top_features = 5 # Default k for comprehensiveness and jaccard

        for i in range(batch_size):
            # --- Existing metrics ---
            auc_del_val = compute_auc_del(x[i], cam_maps[i])
            auc_ins_val = compute_auc_ins(x[i], cam_maps[i])
            
            # --- Stability and Rank Correlation (need perturbed CAMs) ---
            jaccard_sum_for_sample = 0
            rank_corr_sum_for_sample = 0
            num_valid_perturbations_for_stability = 0

            for _ in range(num_perturbations):
                x_perturbed_single = x[i].clone()
                non_pad_mask = (x_perturbed_single != 0)
                non_pad_indices = torch.nonzero(non_pad_mask, as_tuple=False).squeeze()
                
                current_cam_original = cam_maps[i]

                if non_pad_indices.numel() > 0:
                    non_pad_indices = non_pad_indices.view(-1)
                    num_to_perturb = max(1, int(non_pad_indices.size(0) * 0.1))
                    perm = torch.randperm(non_pad_indices.size(0), device=device)
                    indices_to_perturb = non_pad_indices[perm[:num_to_perturb]]
                    x_perturbed_single[indices_to_perturb] = 1  # UNK token

                with torch.no_grad():
                    # Ensure x_perturbed_single is LongTensor for embedding layer
                    cam_perturbed_single = self.grad_cam_auto(x_perturbed_single.unsqueeze(0).long())[0]
                
                # Jaccard
                k_orig = min(k_top_features, current_cam_original.numel())
                k_pert = min(k_top_features, cam_perturbed_single.numel())
                if k_orig > 0 and k_pert > 0:
                    _, top_k_orig_indices = torch.topk(current_cam_original, k_orig)
                    _, top_k_pert_indices = torch.topk(cam_perturbed_single, k_pert)
                    set_orig = set(top_k_orig_indices.cpu().tolist())
                    set_pert = set(top_k_pert_indices.cpu().tolist())
                    intersection = len(set_orig & set_pert)
                    union = len(set_orig | set_pert)
                    jaccard_val = intersection / union if union > 0 else 0.0
                    jaccard_sum_for_sample += jaccard_val
                    
                    # Rank Correlation
                    rank_corr_val = compute_rank_correlation_single(current_cam_original, cam_perturbed_single)
                    if not np.isnan(rank_corr_val):
                        rank_corr_sum_for_sample += rank_corr_val
                        num_valid_perturbations_for_stability +=1
            
            avg_jaccard_stability = jaccard_sum_for_sample / num_perturbations if num_perturbations > 0 else np.nan
            avg_rank_correlation = rank_corr_sum_for_sample / num_valid_perturbations_for_stability if num_valid_perturbations_for_stability > 0 else np.nan

            # --- Comprehensiveness ---
            comprehensiveness_val = compute_comprehensiveness_single(x[i], cam_maps[i], k=k_top_features)

            # Calculate ECS (Explanation Consistency Score) - using new avg_jaccard_stability
            faithfulness = (auc_ins_val - auc_del_val + 1) / 2  # Normalize to [0,1]
            simplicity = 1 - (torch.count_nonzero(cam_maps[i]) / cam_maps[i].numel()).item()
            # Adjusted ECS weights slightly to include rank correlation if desired, or keep as is.
            # For now, keeping original ECS structure but using the re-calculated Jaccard.
            ecs = 0.5 * faithfulness + 0.4 * avg_jaccard_stability + 0.1 * simplicity

            metrics['auc_del'].append(auc_del_val)
            metrics['auc_ins'].append(auc_ins_val)
            metrics['jaccard_stability'].append(avg_jaccard_stability)
            metrics['comprehensiveness'].append(comprehensiveness_val)
            metrics['rank_correlation'].append(avg_rank_correlation)
            metrics['ecs'].append(ecs)

        # Average metrics across batch
        return {k: np.mean(v) for k, v in metrics.items()}

    def generate_adversarial_example(self, x, y, epsilon=0.1, num_steps=10):
        """
        Generate adversarial examples using the Fast Gradient Sign Method (FGSM)
        Args:
            x: Input tensor (batch_size, seq_len)
            y: Target labels
            epsilon: Maximum perturbation size
            num_steps: Number of optimization steps
        Returns:
            Adversarial examples
        """
        self.train()  # Enable gradients

        # Ensure model and tensors are on the same device
        device = next(self.parameters()).device
        x = x.to(device).long()
        y = y.to(device).float()

        # Store original input for later use
        x_orig = x.clone().detach()

        # Get initial embeddings
        with torch.no_grad():
            embeddings = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # Create adversarial embeddings starting from original embeddings
        emb_adv = embeddings.clone().detach().requires_grad_(True)
        criterion = nn.BCELoss()

        for _ in range(num_steps):
            # Forward pass with current adversarial embeddings
            emb_permuted = emb_adv.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
            x1 = F.relu(self.conv1(emb_permuted))
            x2 = F.relu(self.conv2(x1))
            x3 = F.relu(self.conv3(x2))
            pooled = self.global_max_pool(x3).squeeze(-1)
            x_fc1 = self.dropout(F.relu(self.fc1(pooled)))
            logits = self.fc2(x_fc1)
            outputs = torch.sigmoid(logits)  # Apply sigmoid before BCE loss

            # Compute loss
            loss = criterion(outputs.squeeze(), y.float())

            # Compute gradients
            loss.backward()

            # Update adversarial embeddings
            with torch.no_grad():
                # Get gradient sign
                grad_sign = emb_adv.grad.sign()
                # Update embeddings
                emb_adv.add_(epsilon * grad_sign)

                # Project embeddings back to valid space if needed
                if _ < num_steps - 1:  # Don't need to zero grad on last iteration
                    emb_adv.grad.zero_()

        # Now we need to find the closest word indices for our perturbed embeddings
        with torch.no_grad():
            # Get embedding matrix
            emb_matrix = self.embedding.weight  # (vocab_size, embedding_dim)

            # Initialize output tensor
            batch_size, seq_len, emb_dim = emb_adv.size()
            x_adv = torch.zeros_like(x_orig)

            # Process in smaller batches to avoid memory issues
            batch_size_inner = 128  # Process 128 tokens at a time

            for i in range(0, batch_size * seq_len, batch_size_inner):
                # Get current batch of embeddings
                start_idx = i
                end_idx = min(i + batch_size_inner, batch_size * seq_len)

                # Reshape current batch of embeddings
                current_emb = emb_adv.view(-1, emb_dim)[start_idx:end_idx]

                # Compute cosine similarity for current batch
                current_emb_normalized = F.normalize(current_emb, p=2, dim=1)
                emb_matrix_normalized = F.normalize(emb_matrix, p=2, dim=1)

                # Compute similarities batch-wise
                similarities = torch.mm(current_emb_normalized, emb_matrix_normalized.t())

                # Get closest words for current batch
                closest_words = similarities.argmax(dim=1)

                # Place results in the output tensor
                x_adv.view(-1)[start_idx:end_idx] = closest_words

            # Ensure we don't modify padding tokens
            pad_mask = (x_orig == 0)  # Assuming 0 is PAD token
            x_adv = torch.where(pad_mask, x_orig, x_adv)

        self.eval()  # Reset to evaluation mode
        return x_adv

    def measure_adversarial_robustness(self, x, y, epsilon_range=[0.01, 0.05, 0.1]):
        """
        Measure model robustness against adversarial attacks
        Args:
            x: Input tensor
            y: True labels
            epsilon_range: List of perturbation sizes to test
        Returns:
            dict: Dictionary containing robustness metrics
        """
        metrics = {
            'clean_accuracy': None,
            'adversarial_accuracy': {},
            'explanation_shift': {}
        }

        # Ensure model is on CUDA if available
        if torch.cuda.is_available():
            self.cuda()
            device = 'cuda'
        else:
            self.cpu()
            device = 'cpu'

        # Move input tensors to the same device as model and ensure correct type
        x = x.to(device).long()
        y = y.to(device).float()

        # Ensure embedding layer is on the correct device
        self.embedding = self.embedding.to(device)

        # Get clean predictions and explanations
        with torch.no_grad():
            clean_preds = self(x)
            clean_cam = self.grad_cam(x)

        # Calculate clean accuracy (ensure all tensors are on same device)
        clean_preds = clean_preds.to(device)
        clean_acc = ((clean_preds > 0.5).float() == y).float().mean().item()
        metrics['clean_accuracy'] = clean_acc

        # Test different perturbation sizes
        for epsilon in epsilon_range:
            # Generate adversarial examples
            x_adv = self.generate_adversarial_example(x, y, epsilon=epsilon)

            # Get adversarial predictions and explanations
            with torch.no_grad():
                adv_preds = self(x_adv)
                adv_cam = self.grad_cam(x_adv)

            # Calculate adversarial accuracy (ensure all tensors are on same device)
            adv_preds = adv_preds.to(device)
            adv_acc = ((adv_preds > 0.5).float() == y).float().mean().item()
            metrics['adversarial_accuracy'][epsilon] = adv_acc

            # Calculate explanation shift (ensure all tensors are on same device)
            clean_cam = clean_cam.to(device)
            adv_cam = adv_cam.to(device)
            cos_sim = F.cosine_similarity(clean_cam.view(x.size(0), -1),
                                          adv_cam.view(x.size(0), -1), dim=1)
            avg_shift = 1 - cos_sim.mean().item()  # Convert similarity to distance
            metrics['explanation_shift'][epsilon] = avg_shift

        return metrics

    def evaluate_adversarial_examples(self, x, y):
        """
        Comprehensive evaluation of model behavior under adversarial attack
        Args:
            x: Input tensor
            y: True labels
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        self.eval()  # Ensure model is in evaluation mode

        # Standard metrics on clean data
        with torch.no_grad():
            clean_outputs = self(x)
            clean_preds = (clean_outputs > 0.5).float()
            clean_metrics = compute_metrics(y, clean_preds, clean_outputs)

        # Generate adversarial examples
        x_adv = self.generate_adversarial_example(x, y)

        # Metrics on adversarial examples
        with torch.no_grad():
            adv_outputs = self(x_adv)
            adv_preds = (adv_outputs > 0.5).float()
            adv_metrics = compute_metrics(y, adv_preds, adv_outputs)

        # Get explanations for both clean and adversarial
        clean_cam = self.grad_cam_auto(x)
        adv_cam = self.grad_cam_auto(x_adv)

        # Compute explanation metrics for both
        clean_exp_metrics = self.compute_explanation_metrics(x, clean_cam)
        adv_exp_metrics = self.compute_explanation_metrics(x_adv, adv_cam)

        return {
            'clean': {
                'performance': clean_metrics,
                'explanations': clean_exp_metrics
            },
            'adversarial': {
                'performance': adv_metrics,
                'explanations': adv_exp_metrics
            }
        }
