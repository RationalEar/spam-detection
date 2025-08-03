import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple
from utils.device_utils import ensure_device_consistency


def grad_cam(model, x, target_class=None):
    """
    Compute Grad-CAM for the input batch x.
    Args:
        model: CNN model
        x: input tensor (batch_size, seq_len)
        target_class: index of the class to compute Grad-CAM for (default: predicted class)
    Returns:
        cam: class activation map (batch_size, seq_len)
    """
    model.eval()

    # Store original device and ensure consistency
    original_device = next(model.parameters()).device

    # Ensure model is on the same device as input
    input_device = x.device
    working_device = input_device if input_device.type in ['cuda', 'cpu'] else original_device

    # Move model to working device and ensure consistency
    ensure_device_consistency(model, working_device)

    # Ensure input is on the working device
    x_device = x.detach().to(working_device).long()
    if target_class is not None:
        target_class = target_class.detach().to(working_device)

    # Set up for gradient capture
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())
    
    # Register hooks
    handle_fwd = model.conv3.register_forward_hook(forward_hook)
    handle_bwd = model.conv3.register_full_backward_hook(backward_hook)
    
    try:
        # Forward pass
        with torch.set_grad_enabled(True):
            # Pass through model - ensure x is LongTensor for embedding
            x_emb = model.embedding(x_device)
            x_perm = x_emb.permute(0, 2, 1)
            x1 = F.relu(model.conv1(x_perm))
            x2 = F.relu(model.conv2(x1))
            x3 = F.relu(model.conv3(x2))
            pooled = model.global_max_pool(x3).squeeze(-1)
            x_fc1 = model.dropout(F.relu(model.fc1(pooled)))
            logits = model.fc2(x_fc1)
            
            # For binary classification
            if target_class is None:
                # Just use logits for gradient
                loss = logits.sum()
            else:
                # Target-specific loss
                loss = (logits * target_class.float()).sum()
            
            # Compute gradients
            model.zero_grad()
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
        
        # Move result to the same device as original input
        result_cam = cam.to(input_device)

        return result_cam

    except Exception as e:
        print(f"Error in grad_cam: {str(e)}")
        # Return uniform importance as fallback on the same device as input
        fallback = torch.ones(x.size(0), x.size(1), dtype=torch.float, device=input_device)
        return fallback

    finally:
        # Always remove hooks
        handle_fwd.remove()
        handle_bwd.remove()
        # Restore model to its original device and ensure consistency
        ensure_device_consistency(model, original_device)


def grad_cam_auto(model, x, target_class=None):
    """
    CUDA-based Grad-CAM implementation.
    Computes class activation maps for the input batch x using the final convolutional layer.
    
    Args:
        model: CNN model
        x: input tensor (batch_size, seq_len)
        target_class: index of the class to compute Grad-CAM for (default: predicted class)
    Returns:
        cam: class activation map (batch_size, seq_len)
    """
    model.eval()  # Ensure model is in evaluation mode
    
    # Ensure model and input are on CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This implementation requires CUDA.")
    
    model.cuda()  # Move model to CUDA
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
    handle_fwd = model.conv3.register_forward_hook(save_activation)
    handle_bwd = model.conv3.register_full_backward_hook(save_gradient)
    
    try:
        # Forward pass with gradient computation
        with torch.set_grad_enabled(True):
            # Ensure input is CUDA LongTensor
            if not isinstance(x, torch.cuda.LongTensor):
                x = x.long().cuda()
            
            # Get embeddings
            x_emb = model.embedding(x)  # (batch_size, seq_len, embedding_dim)
            x_emb = x_emb.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
            
            # Convolutional layers
            x1 = F.relu(model.conv1(x_emb))
            x2 = F.relu(model.conv2(x1))
            x3 = F.relu(model.conv3(x2))
            
            # Global max pooling and final layers
            pooled = model.global_max_pool(x3).squeeze(-1)
            x_fc1 = model.dropout(F.relu(model.fc1(pooled)))
            logits = model.fc2(x_fc1)
            
            if target_class is None:
                # Just use logits for gradient
                loss = logits.sum()
            else:
                # Target-specific loss
                loss = (logits * target_class.float()).sum()
            
            # Backward pass
            model.zero_grad(set_to_none=True)
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
        model.zero_grad(set_to_none=True)


def grad_cam_with_timing(model, x, target_class=None, return_timing=True):
    """
    Compute Grad-CAM for the input batch x with timing measurements.
    Args:
        model: CNN model
        x: input tensor (batch_size, seq_len)
        target_class: index of the class to compute Grad-CAM for (default: predicted class)
        return_timing: whether to return timing information
    Returns:
        If return_timing=True:
            tuple: (cam, timing_info) where timing_info is a dict with timing details
        If return_timing=False:
            cam: class activation map (batch_size, seq_len)
    """
    timing_info = {}
    total_start = time.time()
    
    # Time the actual grad_cam computation
    grad_cam_start = time.time()
    cam = grad_cam(model, x, target_class)
    grad_cam_end = time.time()
    
    total_end = time.time()
    
    # Record timing information
    timing_info = {
        'grad_cam_time': grad_cam_end - grad_cam_start,
        'total_time': total_end - total_start,
        'batch_size': x.size(0),
        'seq_length': x.size(1),
        'time_per_message': (grad_cam_end - grad_cam_start) / x.size(0)
    }
    
    if return_timing:
        return cam, timing_info
    else:
        return cam


def measure_grad_cam_timing(model, messages: List[torch.Tensor], target_classes=None,
                           batch_size=1) -> Dict[str, List[float]]:
    """
    Measure timing for Grad-CAM explanation generation across multiple messages.
    Args:
        model: CNN model
        messages: List of input tensors, each representing a message
        target_classes: List of target classes (optional)
        batch_size: Size of batches to process (default: 1 for per-message timing)
    Returns:
        Dict with timing statistics per message and overall statistics
    """
    timing_results = {
        'per_message_times': [],
        'batch_times': [],
        'message_lengths': [],
        'batch_sizes': []
    }
    
    # Process messages in batches
    for i in range(0, len(messages), batch_size):
        batch_end = min(i + batch_size, len(messages))
        batch_messages = messages[i:batch_end]
        
        # Stack messages into a batch tensor
        # Pad to same length if necessary
        max_len = max(msg.size(-1) for msg in batch_messages)
        padded_messages = []
        for msg in batch_messages:
            if msg.dim() == 1:
                msg = msg.unsqueeze(0)  # Add batch dimension if missing
            if msg.size(-1) < max_len:
                # Pad with zeros
                pad_size = max_len - msg.size(-1)
                msg = torch.cat([msg, torch.zeros(msg.size(0), pad_size, dtype=msg.dtype)], dim=-1)
            padded_messages.append(msg)
        
        batch_tensor = torch.cat(padded_messages, dim=0)
        
        # Get target classes for this batch if provided
        batch_target = None
        if target_classes is not None:
            batch_target = target_classes[i:batch_end]
            if isinstance(batch_target, list):
                batch_target = torch.tensor(batch_target)
        
        # Measure timing for this batch
        batch_start = time.time()
        cam, timing_info = grad_cam_with_timing(model, batch_tensor, batch_target, return_timing=True)
        batch_end_time = time.time()
        
        # Record timing information
        timing_results['batch_times'].append(timing_info['total_time'])
        timing_results['batch_sizes'].append(batch_tensor.size(0))
        
        # Record per-message timing (approximate for batches > 1)
        for j in range(len(batch_messages)):
            timing_results['per_message_times'].append(timing_info['time_per_message'])
            timing_results['message_lengths'].append(batch_messages[j].size(-1))
    
    # Calculate summary statistics
    timing_results['stats'] = {
        'mean_time_per_message': sum(timing_results['per_message_times']) / len(timing_results['per_message_times']),
        'min_time_per_message': min(timing_results['per_message_times']),
        'max_time_per_message': max(timing_results['per_message_times']),
        'total_messages': len(timing_results['per_message_times']),
        'total_time': sum(timing_results['batch_times'])
    }
    
    return timing_results


def print_timing_summary(timing_results: Dict[str, List[float]]):
    """
    Print a summary of timing results in a readable format.
    """
    stats = timing_results['stats']
    
    print("=== Grad-CAM Timing Summary ===")
    print(f"Total messages processed: {stats['total_messages']}")
    print(f"Total time: {stats['total_time']:.4f} seconds")
    print(f"Mean time per message: {stats['mean_time_per_message']:.4f} seconds")
    print(f"Min time per message: {stats['min_time_per_message']:.4f} seconds")
    print(f"Max time per message: {stats['max_time_per_message']:.4f} seconds")
    print(f"Messages per second: {stats['total_messages'] / stats['total_time']:.2f}")
    
    # Message length analysis
    if timing_results['message_lengths']:
        import statistics
        lengths = timing_results['message_lengths']
        times = timing_results['per_message_times']
        
        print(f"\n=== Message Length Analysis ===")
        print(f"Mean message length: {statistics.mean(lengths):.1f} tokens")
        print(f"Min message length: {min(lengths)} tokens")
        print(f"Max message length: {max(lengths)} tokens")
        
        # Correlation between length and time (simple)
        if len(lengths) > 1:
            correlation = statistics.correlation(lengths, times) if hasattr(statistics, 'correlation') else 0
            print(f"Length-Time correlation: {correlation:.3f}")


def visualize_explanation(text, cam_map, pred_prob, idx, max_len):
    # Tokenize text
    tokens = text.split()[:max_len]  # Truncate to max_len
    
    # Create heatmap
    plt.figure(figsize=(15, 3))
    plt.imshow(cam_map[:len(tokens)].cpu().numpy().reshape(1, -1),
               aspect='auto', cmap='hot')
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.colorbar(label='Importance')
    plt.title(f'Explanation for Sample {idx} (Pred: {"Spam" if pred_prob > 0.5 else "Ham"}, '
              f'Confidence: {pred_prob:.2f})')
    plt.tight_layout()
    plt.show()
