import torch   
import torch.nn as nn


class HilbertScan(nn.Module):
    """
    Reorders pixels according to a Hilbert Curve for multi-channel images.
    
    The Hilbert curve is a space-filling curve that preserves spatial locality
    when mapping 2D coordinates to 1D sequences. This module applies the same
    Hilbert curve pattern to each channel independently, then reorganizes the
    output so the sequence dimension comes first.
    
    Supports grayscale (C=1) or RGB (C=3) images.
    
    Attributes
    ----------
    indices : torch.LongTensor
        Precomputed Hilbert curve indices for an n×n grid, stored as a
        non-trainable buffer.
    
    Input
    -----
    x : torch.Tensor
        Input tensor of shape (B, C, H, W), where
        B : batch size
        C : number of channels
        H : height (n)
        W : width (n)
    
    Returns
    -------
    out : torch.Tensor
        Reordered tensor of shape (B, seq_len, C) where seq_len = H*W = n*n.
        Pixels are arranged according to the Hilbert curve traversal order.
    """
    def __init__(self, n=64):
        """Initialize HilbertScan with precomputed indices for an n x n grid.

        Parameters
        ----------
        n : int, optional
            Grid size (must be a power of 2). Default 64, which keeps the
            existing GalaxyClassifierS4D baseline (scanning the raw 64x64
            image) unaffected. The hybrid CNN+S4D classifier passes a
            smaller n (e.g. 16) since it scans the CNN stem's downsampled
            feature map instead of the raw image.
        """
        super().__init__()
        self.n = n
        indices = self.get_hilbert_indices(n)
        self.register_buffer('indices', indices)

    def _rot(self, s, x, y, rx, ry):
    
        if ry == 0:                  # Bottom half of the current square
            if rx == 1:              # Bottom-right quadrant
                x = s - 1 - x        # Reflect over diagonal
                y = s - 1 - y
            x, y = y, x              # Swap x and y for 90° rotation
        return x, y


    def _d2xy(self, n, d):
        """
        Convert 1D Hilbert curve distance to 2D coordinates.
        
        This implements the Hilbert curve mapping algorithm that converts
        a linear distance along the curve to (x, y) coordinates.
        
        Parameters
        ----------
        n : int
            Size of the grid (must be a power of 2).
        d : int
            Distance along the Hilbert curve (0 to n²-1).
        
        Returns
        -------
        tuple of int
            (x, y) coordinates in the grid.
        """
        x = 0
        y = 0
        t = d
        s = 1
#Determine which quadrant of the current square this distance is in
        while s < n:
            rx = (t // 2) & 1
            ry = (t ^ rx) & 1
#Rotate and/or reflect coordinates depending on quadrant
            x, y = self._rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
#Move to next level of recursion (divide distance by 4 for next smaller square)
            t //= 4
            s *= 2

        return x, y

    def get_hilbert_indices(self, n):
        """
        Generate Hilbert curve indices for an n x n grid.
        
        Creates a lookup table that maps Hilbert curve positions to
        flattened array indices for a 2D grid.
        
        Parameters
        ----------
        n : int
            Grid size (must be a power of 2).
        
        Returns
        -------
        torch.LongTensor
            Tensor of shape (n²,) containing flattened indices following
            the Hilbert curve traversal order.
        """
        indices = []
        for d in range(n * n):
            x, y = self._d2xy(n, d)
            # GalaxyMNIST is 64x64, power of 2
            if x < n and y < n:
                indices.append(y * n + x)
        return torch.LongTensor(indices)

    def forward(self, x):
        """
        Apply Hilbert curve reordering to input images.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (B, C, H, W).
        
        Returns
        -------
        torch.Tensor
            Reordered tensor of shape (B, seq_len, C) where seq_len = H*W,
            with pixels arranged in Hilbert curve order.
        """
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.view(B, C, -1)           # Flatten each channel: (B, C, H*W)
        x = x[:, :, self.indices]      # Reorder according to Hilbert: (B, C, H*W)
        x = x.permute(0, 2, 1)         # (B, seq_len, C) so sequence dimension is 1D
        return x
if __name__ == "__main__":
    import torch

    img = torch.arange(64*64).view(1,1,64,64).float()
    hilbert = HilbertScan()
    out = hilbert(img)

    print(out[0, :20, 0])