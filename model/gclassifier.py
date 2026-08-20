import torch
import torch.nn as nn

from .hilbert import HilbertScan
from .tlts import TakeLastTimestep
from .s4d_recurrent import S4D

class GalaxyClassifierS4D(nn.Module):
    """
    Galaxy classifier using Hilbert Scan and S4 sequence modeling.
    
    This model scans 2D galaxy images into a 1D Hilbert sequence, projects
    the multi-channel pixel values to a higher-dimensional feature space,
    processes the sequence with stacked S4 layers with GELU activations, 
    takes the final timestep as a summary representation, and applies a 
    linear classifier to predict galaxy types.
    
    Parameters
    ----------
    s4_state : int, optional
        Hidden state dimension for the S4 layers (default is 64).
    d_model : int, optional
        Output feature dimension of the S4 layers (default is 64).
    num_classes : int, optional
        Number of output classes (default is 4).
    colored : bool, optional
        If True, expects RGB input images (3 channels); if False, expects
        grayscale images (1 channel) (default is True).
    
    Attributes
    ----------
    seq_len : int
        Sequence length after Hilbert scan (64*64 = 4096).
    d_model : int
        Dimension of the S4 output features.
    hilbert_channels : int
        Number of input channels (1 for grayscale, 3 for RGB).
    hilbert_scan : HilbertScan
        Layer that converts 2D images into 1D sequences using a Hilbert scan.
    uproject : nn.Linear
        Linear projection mapping hilbert_channels to d_model dimensions.
    s4_1 : S4D
        First S4 layer.
    act1 : nn.GELU
        GELU activation after the first S4 layer.
    s4_2 : S4D
        Second S4 layer.
    act2 : nn.GELU
        GELU activation after the second S4 layer.
    take_last : TakeLastTimestep
        Layer that extracts the last timestep from the sequence.
    fc : nn.Linear
        Linear classifier mapping S4 features to output classes.
    softmax : nn.Softmax
        Softmax layer for output probabilities.
    """
    def __init__(self, s4_state=64, d_model=64, num_classes=4, colored=True):
        super().__init__()
        self.seq_len = 64 * 64 
        self.d_model = d_model

        # Hilbert Scan layer
        self.hilbert_scan = HilbertScan()
        self.hilbert_channels = 1 if not colored else 3

        self.uproject = nn.Linear(self.hilbert_channels, d_model)

        # S4 layers -- recurrent, not the old FFT/causal-conv layer. Verified
        # against the trained portable first (see recurrent_vs_causal_conv_verification.png):
        # logits matched to ~6e-4, same argmax on every sample. Conv layer's gone now.
        self.s4_1 = S4D(d_model=d_model, d_state=s4_state, transposed=False)
        self.act1 = nn.GELU()

        self.s4_2 = S4D(d_model=d_model, d_state=s4_state, transposed=False)
        self.act2 = nn.GELU()

        # Take last timestep
        self.take_last = TakeLastTimestep()

        # Classifier
        self.fc = nn.Linear(d_model, num_classes)

        # Softmax for output probabilities
        self.softmax = nn.Softmax(dim=-1)


       # -------------------------------------------------------------------------
        # PARAMETER COUNT VERIFICATION (Task 8.4)
        # Verified with torchinfo.summary()
        # -------------------------------------------------------------------------
        # 1. Input Projection: (1 * 64) + 64 = 128 params
        # 2. S4D Layer 1 (Optimized N/2 symmetry): 
        #    Per feature: 130 params (vs 258 naive)
        #    Total: 64 * 130 = 8,320 params
        # 3. S4D Layer 2: Same as Layer 1 = 8,320 params
        # 4. Classifier Head: (64 * 4) + 4 = 260 params
        # 
        # GRAND TOTAL: 128 + 8,320 + 8,320 + 260 = 17,028 Parameters
        # -------------------------------------------------------------------------



        # -------------------------------------------------------------------------
        # FLOPS ESTIMATION (Task 8.5) -- redone for the recurrent S4D layer
        # Sequence Length L = 4096, d_model = 64, d_state = 64, C = 1
        # -------------------------------------------------------------------------
        # 1. Input Projection: L * C * d_model
        #    4096 * 1 * 64 = 262,144 Ops
        #
        # 2. S4D Layers (x2): no more FFT kernel, so no log(L) term. Each layer
        #    steps through L timesteps, and at each step does ~2 complex MACs per
        #    state element (one for the state update, one for the output sum) --
        #    a complex MAC costs roughly 4x a real one, call it ~8 real ops:
        #    2 * (L * (d_state/2) * d_model * 8) = 2 * (4096*32*64*8) ≈ 134.2M Ops
        #
        # 3. Classifier Head: d_model * Classes
        #    64 * 4 = 256 Ops
        #
        # GRAND TOTAL: ~134.5 Million Operations per forward pass
        # (vs. ~6.55M under the old FFT estimate -- more raw arithmetic, since
        # we lost the O(log L) speedup, but no transcendental-heavy kernel
        # generation either, which is most of why it still benchmarks faster
        # in practice at this d_model -- see model/s4d_recurrent.py)
        # -------------------------------------------------------------------------

    def forward(self, x, return_logits=False):
        """
        Forward pass of the PixelS4Galaxy model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, 64, 64), where B is the batch size
            and C is the number of channels (1 for grayscale, 3 for RGB).
        return_logits : bool, optional
            If True, returns raw logits instead of softmax probabilities 
            (default is False).
        
        Returns
        -------
        output : torch.Tensor
            If return_logits=True: Output logits of shape (B, num_classes),
            representing unnormalized scores for each galaxy class.
            If return_logits=False: Output probabilities of shape (B, num_classes),
            representing the softmax probability distribution over classes.
        """
        B, C, H, W = x.shape
        assert H == 64 and W == 64, "Expected 64x64"
        assert C == self.hilbert_channels, f"Expected {self.hilbert_channels} channels"

        # 1. Hilbert scan: 2D > 1D
        x_seq = self.hilbert_scan(x)  # (B,4096,C)

        # 2. Input projection: C > d_model
        x_proj = self.uproject(x_seq)  # (B,4096,d_model)

        # 3. S4D layer 1 + GELU
        s4_out1, _ = self.s4_1(x_proj)
        a1 = self.act1(s4_out1)  # (B,4096,d_model

        # 4. S4D layer 2 + GELU
        s4_out2, _ = self.s4_2(a1)
        a2 = self.act2(s4_out2)      # (B,4096,d_model)

        # 5. Take last timestep
        last = self.take_last(a2)          # (B,d_model)

        # 6. Classifier: d_model > num_classes
        logits = self.fc(last)             # (B,4)

        # Return logits or softmax
        if return_logits:
            return logits
        return self.softmax(logits)

# basically this function takes the image and turns it into a sequence
        # first we check the shape to make sure its 64x64
        # then the hilbert scan flattens the 2D image into a long 1D list of pixels
        # after that we project it up to hidden size using a linear layer
        # then it goes through two S4 layers with GELU activation in between to learn features
        # since its a sequence model we only care about the very last timestep which has the summary
        # finally we pass that last step to the linear classifier to get the 4 class scores
        # and if we need probs we apply softmax otherwise just return the raw logits

        #raise NotImplementedError("Forward method not implemented yet.")