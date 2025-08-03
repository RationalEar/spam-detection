import torch


def ensure_device_consistency(model, force_device=None):
    """
    Ensure all model components are on the same device.
    
    Args:
        model: PyTorch model
        force_device: Device to force model to (None for auto-detection)
    
    Returns:
        device: The device the model is now on
    """
    # Determine target device
    if force_device is not None:
        device = torch.device(force_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Move model to device
    model = model.to(device)
    
    # Explicitly move all parameters
    for param in model.parameters():
        param.data = param.data.to(device)
        if param.grad is not None:
            param.grad = param.grad.to(device)
    
    # Explicitly move all buffers
    for buffer in model.buffers():
        buffer.data = buffer.data.to(device)
    
    # Special handling for embedding layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight = torch.nn.Parameter(module.weight.to(device))
    
    return device


def reset_model_device(model, target_device=None):
    """
    Reset model to ensure all components are on the same device.
    This is a more aggressive approach that recreates the model state.

    Args:
        model: PyTorch model
        target_device: Target device ('cuda' or 'cpu', None for auto-detection)

    Returns:
        device: The device the model is now on
    """
    if target_device is None:
        target_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(target_device)

    # Get the current state dict
    state_dict = model.state_dict()

    # Move all state dict tensors to target device
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device)

    # Load the state dict back (this ensures consistency)
    model.load_state_dict(state_dict)

    # Move model to device
    model.to(device)

    # Extra safety: ensure embedding layer is properly on device
    if hasattr(model, 'embedding'):
        model.embedding.weight.data = model.embedding.weight.data.to(device)

    return device


def safe_model_call(model, x, ensure_consistency=True):
    """
    Safely call a model ensuring device consistency.
    
    Args:
        model: PyTorch model
        x: Input tensor
        ensure_consistency: Whether to ensure device consistency first
    
    Returns:
        Model output
    """
    if ensure_consistency:
        device = ensure_device_consistency(model)
        x = x.to(device)
    
    return model(x)


class DeviceSafeModelWrapper:
    """
    A wrapper that ensures device consistency for model calls.
    """
    
    def __init__(self, model):
        self.model = model
        self.device = ensure_device_consistency(model)
    
    def __call__(self, x):
        x = x.to(self.device)
        return self.model(x)
    
    def __getattr__(self, name):
        # Delegate all other attributes to the wrapped model
        return getattr(self.model, name)
