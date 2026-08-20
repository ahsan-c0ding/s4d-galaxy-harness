import math
import torch
import torch.nn as nn
from einops import repeat

class S4D(nn.Module):
    """
    Diagonal Structured State Space (S4D) layer.
    
    Implements the S4D variant of Structured State Spaces using diagonal state matrices
    for computational efficiency. This layer processes sequences through a continuous-time
    state space model discretized using the bilinear method, enabling modeling of long-range
    dependencies with linear complexity.
    
    The S4D model parameterizes the state space with:
    - Diagonal complex-valued state transition matrix A
    - Complex-valued output projection matrix C  
    - Skip connection parameter D
    - Learnable discretization timestep dt
    
    Convolution is performed efficiently in the frequency domain using FFT.
    
    Parameters
    ----------
    d_model : int
        Input and output feature dimension (number of independent SSM copies).
    d_state : int, optional
        Latent state dimension (must be even for complex representation). 
        Default is 64.
    dt_min : float, optional
        Minimum discretization timestep. Default is 0.001.
    dt_max : float, optional
        Maximum discretization timestep. Default is 0.1.
    transposed : bool, optional
        If True, expects input shape (B, H, L). If False, expects (B, L, H).
        Default is True.
    lr : float, optional
        Custom learning rate for SSM parameters. If None, uses optimizer default.
        If 0.0, parameters become fixed buffers.
    
    Attributes
    ----------
    h : int
        Number of independent SSM copies (equals d_model).
    n : int
        State dimension.
    log_dt : nn.Parameter or buffer
        Log-space discretization timestep (shape: h).
    log_A_real : nn.Parameter or buffer
        Log-space real part of diagonal state matrix (shape: h, n//2).
    A_imag : nn.Parameter or buffer
        Imaginary part of diagonal state matrix (shape: h, n//2).
    C : nn.Parameter
        Complex output projection matrix (shape: h, n//2, 2 for real view).
    D : nn.Parameter  
        Skip connection weights (shape: h).
    
    Input
    -----
    u : torch.Tensor
        Input sequence of shape (B, H, L) if transposed=True, or (B, L, H) otherwise.
        B : batch size
        H : d_model (feature dimension)
        L : sequence length
    
    Returns
    -------
    y : torch.Tensor
        Output sequence of same shape as input.
    None
        Placeholder for compatibility with stateful interfaces.
    
    References
    ----------
    Gu, A., Goel, K., & Ré, C. (2022). Efficiently Modeling Long Sequences with 
    Structured State Spaces. In ICLR 2022.
    
    Gu, A., Gupta, A., Goel, K., & Ré, C. (2022). On the Parameterization and 
    Initialization of Diagonal State Space Models. In NeurIPS 2022.
    """
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1, transposed=True, lr=None):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.transposed = transposed

        # --- Initial Parameter Tensors ---
        log_dt = torch.rand(self.h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        log_A_real = torch.log(0.5 * torch.ones(self.h, self.n // 2))
        A_imag = math.pi * repeat(torch.arange(self.n // 2), 'n -> h n', h=self.h)
        C_init = torch.randn(self.h, self.n // 2, dtype=torch.cfloat)

        # --- Registration ---
        # We use 'register' to set weight_decay=0.0 and custom LRs for SSM cores
        self.register("log_dt", log_dt, lr)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)
        
        # C and D are usually treated as standard parameters
        self.C = nn.Parameter(torch.view_as_real(C_init))
        self.D = nn.Parameter(torch.randn(self.h))

    def register(self, name, tensor, lr=None):
        """
        Register a parameter or buffer with custom optimization settings.
        
        Parameters
        ----------
        name : str
            Name for the parameter/buffer.
        tensor : torch.Tensor
            Tensor to register.
        lr : float, optional
            Custom learning rate. If 0.0, registers as buffer (non-trainable).
            If None, uses optimizer default. Otherwise, attaches custom lr metadata.
        """
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            # Tag the parameter with optimization constraints
            optim = {"weight_decay": 0.0}
            if lr is not None: 
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def forward(self, u):
        """
        Forward pass through the S4D layer.
        
        Computes the convolution of the input sequence with the SSM kernel using FFT.
        The kernel is generated from the continuous-time SSM parameters and discretized
        using the learned timestep dt.
        
        Process:
        1. Materialize SSM parameters (dt, A, C) from log-space representations
        2. Generate discrete convolution kernel K via truncated power series
        3. Perform FFT-based convolution: y = K * u
        4. Add skip connection: y = y + D * u
        
        Parameters
        ----------
        u : torch.Tensor
            Input sequence of shape (B, H, L) if transposed=True, else (B, L, H).
            B : batch size
            H : feature dimension (d_model)
            L : sequence length
        
        Returns
        -------
        y : torch.Tensor
            Output sequence of same shape as input.
        None
            Placeholder for state (included for interface compatibility).
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # 1. Materialize Parameters
        dt = torch.exp(self.log_dt) 
        C = torch.view_as_complex(self.C) 
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag 

        # 2. Generate Kernel K (Diagonal SSM formula)
        dtA = A * dt.unsqueeze(-1)  
        # Power series generation: exp(A * dt * t)
        K_exp = torch.exp(dtA.unsqueeze(-1) * torch.arange(L, device=u.device)) 
        C_tilde = C * (torch.exp(dtA) - 1.) / A
        k = 2 * torch.einsum('hn, hnl -> hl', C_tilde, K_exp).real 

        # 3. FFT Convolution (y = k * u)
        k_f = torch.fft.rfft(k, n=2*L) 
        u_f = torch.fft.rfft(u, n=2*L) 
        y = torch.fft.irfft(u_f * k_f, n=2*L)[..., :L] 

        # 4. Skip Connection
        y = y + u * self.D.unsqueeze(-1)

        if not self.transposed: y = y.transpose(-1, -2)
        return y, None