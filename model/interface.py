import torch

class ModelInterface:
    """
    Unified interface for galaxy classification models.
    
    This class abstracts the implementation details (Python vs RISC-V) and provides
    a consistent API for model inference regardless of backend.
    
    Parameters
    ----------
    implementation : str
        Either 'python' or 'riscv'.
    model_path : str
        Path to model weights (used for Python implementation).
    num_classes : int
        Number of output classes.
    colored : bool
        Whether model expects colored or grayscale images.
    device : torch.device
        Device for inference.
    
    Methods
    -------
    __call__(x)
        Run inference on input tensor x.
    """
    
    def __init__(self, implementation, model_path, num_classes, colored, device):
        """Initialize model based on implementation type."""
        self.implementation = implementation
        self.device = device
        
        if implementation == 'python':
            from model import GalaxyClassifierS4D
            print(f"Loading Python model from {model_path}")
            self.model = GalaxyClassifierS4D(colored=colored).to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            self.model.eval()
            
        elif implementation == 'riscv':
            print("Initializing RISC-V interface")
            # TODO: Setup RISC-V communication/configuration
            self.model = None
    
    def __call__(self, x):
        """
        Run model inference.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        
        Returns
        -------
        torch.Tensor
            Model predictions.
        """
        if self.implementation == 'python':
            return self.model(x)
        elif self.implementation == 'riscv':
            # TODO: Send x to RISC-V QEMU, receive predictions
            raise NotImplementedError("RISC-V inference not yet implemented")
    
    def eval(self):
        """Set model to evaluation mode (for consistency with PyTorch API)."""
        if self.implementation == 'python':
            self.model.eval()
