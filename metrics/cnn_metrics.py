import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr  # For Rank Correlation

import metrics.metrics


def compute_metrics(y_true, y_pred, y_prob=None, fp_cost=0.3, fn_cost=0.7):
    """
    Compute comprehensive evaluation metrics including AUC-ROC, FPR, FNR,
    Accuracy, Precision, Recall, F1-Score, Specificity (Ham Preservation Rate),
    Youden's J, and cost-sensitive evaluation.
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities (for AUC-ROC)
        fp_cost: Cost weight for false positives
        fn_cost: Cost weight for false negatives
    Returns:
        dict: Dictionary containing all metrics
    """
    return metrics.compute_metrics(y_true, y_pred, y_prob, fp_cost, fn_cost)


def compute_explanation_metrics(model, x, cam_maps, num_perturbations=10):
    """
    Compute explainability metrics including faithfulness and stability.
    Args:
        model: The SpamCNN model instance
        x: Input tensor (batch_size, seq_len)
        cam_maps: Grad-CAM activation maps (batch_size, seq_len)
        num_perturbations: Number of perturbations for stability testing
    Returns:
        dict: Dictionary containing explainability metrics
    """
    model.eval()

    # Ensure model is on CUDA if available
    if torch.cuda.is_available():
        model.cuda()
        device = 'cuda'
    else:
        model.cpu()
        device = 'cpu'

    # Move input tensors to the same device as model
    x = x.to(device)
    cam_maps = cam_maps.to(device)

    # Ensure embedding layer is on the correct device
    model.embedding = model.embedding.to(device)

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
            base_pred = model(x_input).item()
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
                pred = model(x_input)
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
            base_pred = model(x_input).item()
        insertions.append(base_pred)

        # Progressively insert most important tokens
        for i in range(1, len(indices)):
            # Reveal the top-i tokens
            x_perturbed = torch.ones_like(x_single).long() * pad_token
            x_perturbed[indices[:i]] = x_single[indices[:i]]

            with torch.no_grad():
                x_input = x_perturbed.unsqueeze(0)
                pred = model(x_input)
            insertions.append(pred.item())

        # Calculate AUC
        auc_ins = np.trapz(insertions) / len(indices)
        return auc_ins

    def compute_comprehensiveness_single(x_single, cam_single, k=5):
        """Computes comprehensiveness for a single sample."""
        x_single = x_single.to(device)
        cam_single = cam_single.to(device)

        pad_token = 0  # Assuming 0 is the PAD token
        num_features = x_single.size(0)
        actual_k = min(k, num_features)
        if actual_k == 0: return 0.0

        with torch.no_grad():
            original_pred = model(x_single.unsqueeze(0).long()).item()

        _, top_k_indices = torch.topk(cam_single, actual_k)

        x_masked = x_single.clone()
        x_masked[top_k_indices] = pad_token

        with torch.no_grad():
            pred_after_removal = model(x_masked.unsqueeze(0).long()).item()

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
            return np.nan  # Not enough data points

        correlation, _ = spearmanr(cam_original_valid, cam_perturbed_valid)
        return correlation

    # Calculate metrics for each sample in batch
    metrics = {
        'auc_del': [],
        'auc_ins': [],
        'jaccard_stability': [],  # Renamed from 'stability'
        'comprehensiveness': [],
        'rank_correlation': [],
        'ecs': []  # Explanation Consistency Score
    }

    k_top_features = 5  # Default k for comprehensiveness and jaccard

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
                cam_perturbed_single = model.grad_cam(x_perturbed_single.unsqueeze(0).long())[0]

            # Jaccard
            k_orig = min(k_top_features, current_cam_original.numel())
            k_pert = min(k_top_features, cam_perturbed_single.numel())
            if k_orig > 0 and k_pert > 0:
                _, top_k_orig_indices = torch.topk(current_cam_original.flatten(), k_orig)
                _, top_k_pert_indices = torch.topk(cam_perturbed_single.flatten(), k_pert)
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
                    num_valid_perturbations_for_stability += 1

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


def generate_adversarial_example(model, x, y, epsilon=0.1, num_steps=10):
    """
    Generate adversarial examples using the Fast Gradient Sign Method (FGSM)
    Args:
        model: The SpamCNN model instance
        x: Input tensor (batch_size, seq_len)
        y: Target labels
        epsilon: Maximum perturbation size
        num_steps: Number of optimization steps
    Returns:
        Adversarial examples
    """
    model.train()  # Enable gradients

    # Ensure model and tensors are on the same device
    device = next(model.parameters()).device
    x = x.to(device).long()
    y = y.to(device).float()

    # Store original input for later use
    x_orig = x.clone().detach()

    # Get initial embeddings
    with torch.no_grad():
        embeddings = model.embedding(x)  # (batch_size, seq_len, embedding_dim)

    # Create adversarial embeddings starting from original embeddings
    emb_adv = embeddings.clone().detach().requires_grad_(True)
    criterion = torch.nn.BCELoss()

    for _ in range(num_steps):
        # Forward pass with current adversarial embeddings
        emb_permuted = emb_adv.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        x1 = F.relu(model.conv1(emb_permuted))
        x2 = F.relu(model.conv2(x1))
        x3 = F.relu(model.conv3(x2))
        pooled = model.global_max_pool(x3).squeeze(-1)
        x_fc1 = model.dropout(F.relu(model.fc1(pooled)))
        logits = model.fc2(x_fc1)
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
        emb_matrix = model.embedding.weight  # (vocab_size, embedding_dim)

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

    model.eval()  # Reset to evaluation mode
    return x_adv


def measure_adversarial_robustness(model, x, y, epsilon_range=[0.01, 0.05, 0.1]):
    """
    Measure model robustness against adversarial attacks
    Args:
        model: The SpamCNN model instance
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
        model.cuda()
        device = 'cuda'
    else:
        model.cpu()
        device = 'cpu'

    # Move input tensors to the same device as model and ensure correct type
    x = x.to(device).long()
    y = y.to(device).float()

    # Ensure embedding layer is on the correct device
    model.embedding = model.embedding.to(device)

    # Get clean predictions and explanations
    with torch.no_grad():
        clean_preds = model(x)
        clean_cam = model.grad_cam(x)

    # Calculate clean accuracy (ensure all tensors are on same device)
    clean_preds = clean_preds.to(device)
    clean_acc = ((clean_preds > 0.5).float() == y).float().mean().item()
    metrics['clean_accuracy'] = clean_acc

    # Test different perturbation sizes
    for epsilon in epsilon_range:
        # Generate adversarial examples
        x_adv = generate_adversarial_example(model, x, y, epsilon=epsilon)

        # Get adversarial predictions and explanations
        with torch.no_grad():
            adv_preds = model(x_adv)
            adv_cam = model.grad_cam(x_adv)

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


def evaluate_adversarial_examples(model, x, y):
    """
    Comprehensive evaluation of model behavior under adversarial attack
    Args:
        model: The SpamCNN model instance
        x: Input tensor
        y: True labels
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()  # Ensure model is in evaluation mode

    # Standard metrics on clean data
    with torch.no_grad():
        clean_outputs = model(x)
        clean_preds = (clean_outputs > 0.5).float()
        clean_metrics = compute_metrics(y, clean_preds, clean_outputs)

    # Generate adversarial examples
    x_adv = generate_adversarial_example(model, x, y)

    # Metrics on adversarial examples
    with torch.no_grad():
        adv_outputs = model(x_adv)
        adv_preds = (adv_outputs > 0.5).float()
        adv_metrics = compute_metrics(y, adv_preds, adv_outputs)

    # Get explanations for both clean and adversarial
    clean_cam = model.grad_cam_auto(x)
    adv_cam = model.grad_cam_auto(x_adv)

    # Compute explanation metrics for both
    clean_exp_metrics = compute_explanation_metrics(model, x, clean_cam)
    adv_exp_metrics = compute_explanation_metrics(model, x_adv, adv_cam)

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
