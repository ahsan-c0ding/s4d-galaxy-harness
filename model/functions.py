import os
import numpy as np

# PyTorch
import torch.nn.functional as F

# GalaxyMNIST dataset
from galaxy_mnist import GalaxyMNIST 

def load_data(root: str, download: bool = True, train: bool = True, colored: bool = False):
    """Load and preprocess GalaxyMNIST dataset.
    
    Parameters:
    -----------
    root : str
        Root directory where the dataset is stored or will be downloaded.
    download : bool
        Whether to download the dataset if not present.
    train : bool
        Whether to load the training set (True) or test set (False).
    colored : bool, optional
        Whether to use colored images (3 channels) or grayscale (1 channel).
        (default is False)
           
    Returns:
    --------
    X : torch.Tensor
        Preprocessed images of shape (N, 1, 64, 64) with pixel values in [0, 1] if grayscale,
        or (N, 3, 64, 64) if colored.
    y_onehot : torch.Tensor
        One-hot encoded labels of shape (N, num_classes).
    y : torch.Tensor
        Original labels of shape (N,).
    """
    dataset = GalaxyMNIST(root=root, download=download, train=train)
    print(f"Original Dataset Size: {len(dataset.data)} samples")

    # 1. Extract and process images: Mean across channels, Normalize to [0, 1]
    # Data shape is (N, 3, 64, 64) -> (N, 1, 64, 64)
    X = dataset.data.float()           # convert from uint8 -> float
    if not colored:
        X = X.mean(dim=1, keepdim=True)  # convert to grayscale by averaging channels
    X = X / 255.0                     # normalize to [0, 1]

    # 2. Extract targets and convert to one-hot encoding
    y = dataset.targets.long()
    # One hot encode the labels
    y_onehot = F.one_hot(y).float()
    return X, y_onehot, y

def format_row(values):
    """Formats a list of values, comma separated, even width"""
    return ", ".join(f"{v:10.6f}" if isinstance(v, (float, np.float32, np.float64)) else f"{v}" for v in values)

def export_model_parameters(model, output_dir="galaxy_s4_model_params"):
    """
    General function to export all model parameters AND buffers to .bin and .txt.
    Logic: (A, B, C) -> A blocks of B rows x C cols.
    Follows natural Row-Major storage order.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    txt_path = os.path.join(output_dir, "weights.txt")
    bin_path = os.path.join(output_dir, "weights.bin")

    print(f"--- Exporting Model: {model.__class__.__name__} ---")

    state_dict = model.state_dict()

    with open(txt_path, "w") as f_txt, open(bin_path, "wb") as f_bin:
        for name, tensor in state_dict.items():
            shape = list(tensor.shape)
            print(f"Saving: {name:40} | Shape: {shape}")
            
            data_np = tensor.detach().cpu().contiguous().numpy()
            f_bin.write(data_np.astype(np.float32).tobytes())

            f_txt.write(f"[{name}] Shape: {shape}\n")

            if len(shape) == 0:
                f_txt.write(f"{data_np.item():10.6f}\n\n")

            elif len(shape) == 1:
                f_txt.write(format_row(data_np) + "\n\n")

            elif len(shape) == 2:
                rows, cols = shape
                for r in range(rows):
                    f_txt.write(format_row(data_np[r]) + "\n")
                f_txt.write("\n")

            elif len(shape) == 3:
                A, B, C = shape
                for a in range(A):
                    f_txt.write(f"# Block {a}\n")
                    for b in range(B):
                        row_values = data_np[a, b, :]
                        f_txt.write(format_row(row_values) + "\n")
                    f_txt.write("\n")
            
            elif len(shape) == 4:
                A, B, C, D = shape
                for a in range(A):
                    for b in range(B):
                        f_txt.write(f"# Block {a}, {b}\n")
                        for c in range(C):
                            f_txt.write(format_row(data_np[a, b, c, :]) + "\n")
                        f_txt.write("\n")

            else:
                f_txt.write(format_row(data_np.flatten()) + "\n\n")

    print(f"--- Export Complete. Files located in '{output_dir}' ---")