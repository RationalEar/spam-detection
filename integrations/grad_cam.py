import torch
import torch.nn.functional as F


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
    # Force CPU computation to avoid CUDA errors
    model.to('cpu')

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
    handle_fwd = model.conv3.register_forward_hook(forward_hook)
    handle_bwd = model.conv3.register_full_backward_hook(backward_hook)

    try:
        # Forward pass
        with torch.set_grad_enabled(True):
            # Pass through model
            x_emb = model.embedding(x_cpu)  # x must remain a LongTensor
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

        return cam

    except Exception as e:
        print(f"Error in grad_cam: {str(e)}")
        # Return uniform importance as fallback
        return torch.ones(x_cpu.size(0), x_cpu.size(1), dtype=torch.float)

    finally:
        # Always remove hooks
        handle_fwd.remove()
        handle_bwd.remove()


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
