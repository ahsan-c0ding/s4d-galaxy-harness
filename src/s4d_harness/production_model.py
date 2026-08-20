"""
Production-family architecture: ConvPatchStem/Linear patch embedding + S4D
(FFT-based parallel convolution) sequence layers.

This is the model behind the project's headline result (86.80% test
accuracy, 528,004 params). Extracted verbatim from the code cells of
notebook-best-s4d-model(1).ipynb, which is byte-identical (diffed and
confirmed) to the same class definitions in 8 of the 9 single-run
production notebooks in notebooks/:

    notebook-best-s4d-model(1).ipynb   (source of this file)
    s4d-2-layers.ipynb
    notebook-64x64-dimension.ipynb
    s4d-108d-test.ipynb
    s4d-108d-test(2).ipynb
    s4d-pooling-and-scale-tests.ipynb
    s4d-testing-scan-variants.ipynb
    s4d-width-scan-test.ipynb

(small-linear-s4d-70k-parameters.ipynb carries a slightly earlier variant
of HilbertScan/TakeLastTimestep/S4DConv, named GalaxyClassifierS4D instead
of GalaxyClassifierS4DFast; both produce the same 76,360-param linear+S4D
model reported in the paper, so this is treated as the same architecture.)

Nothing below has been rewritten or "cleaned up" -- it is the actual code
that produced every Master Results Table row in the LaTeX report. The
report-discrepency-testing.ipynb notebook independently re-defines this
same class under the name MainStudyGalaxyClassifier for its controlled
recipe-crossing study; that copy is diffed identical to this one and is
not duplicated here.
"""
import math

import torch
import torch.nn as nn
from einops import repeat


class HilbertScan(nn.Module):
    """Reorders patches of a (B, C, H, W) image along a Hilbert curve."""

    def __init__(self, image_size=64, patch_size=1):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.register_buffer("indices", self._get_hilbert_indices(self.grid_size))

    @staticmethod
    def _rot(s, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        return x, y

    def _d2xy(self, n, d):
        x = y = 0
        t, s = d, 1
        while s < n:
            rx = (t // 2) & 1
            ry = (t ^ rx) & 1
            x, y = self._rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y

    def _get_hilbert_indices(self, grid_size):
        indices = []
        for d in range(grid_size * grid_size):
            x, y = self._d2xy(grid_size, d)
            indices.append(y * grid_size + x)
        return torch.LongTensor(indices)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, self.num_patches, C * p * p)
        return patches[:, self.indices, :]


class TakeLastTimestep(nn.Module):
    def forward(self, x):
        return x[:, -1, :]


class S4DConv(nn.Module):
    """Fast FFT-based parallel convolution S4D layer."""

    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1, transposed=True, lr=None):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.transposed = transposed

        log_dt = torch.rand(self.h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        log_A_real = torch.log(0.5 * torch.ones(self.h, self.n // 2))
        A_imag = math.pi * repeat(torch.arange(self.n // 2), 'n -> h n', h=self.h)
        C_init = torch.randn(self.h, self.n // 2, dtype=torch.cfloat)

        self.register("log_dt", log_dt, lr)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

        self.C = nn.Parameter(torch.view_as_real(C_init))
        self.D = nn.Parameter(torch.randn(self.h))

    def register(self, name, tensor, lr=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def forward(self, u):
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag

        dtA = A * dt.unsqueeze(-1)
        K_exp = torch.exp(dtA.unsqueeze(-1) * torch.arange(L, device=u.device))
        C_tilde = C * (torch.exp(dtA) - 1.) / A
        k = 2 * torch.einsum('hn, hnl -> hl', C_tilde, K_exp).real

        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]
        y = y + u * self.D.unsqueeze(-1)

        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, None


class ConvPatchStem(nn.Module):
    """Convolutional stem for local neighborhood mixing before patch projection.

    Two plain, unnormalized convolutions -- deliberately kept minimal for
    eventual bare-metal RISC-V portability (see the CAAL S4D port project).
    """

    def __init__(self, in_channels, d_model, patch_size):
        super().__init__()
        mid_channels = max(in_channels * 8, 32)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, d_model, kernel_size=patch_size, stride=patch_size),
        )

    def forward(self, x):
        return self.net(x)


class GalaxyClassifierS4DFast(nn.Module):
    """Production S4D Classifier (conv or linear patch embedding + S4D stack)."""

    def __init__(self, s4_state=64, d_model=64, num_classes=4, colored=True,
                 num_layers=2, patch_size=1, pooling="last",
                 use_norm=False, use_residual=False, dropout=0.0,
                 patch_embed="linear"):
        super().__init__()
        self.hilbert_channels = 1 if not colored else 3
        self.patch_size = patch_size
        self.pooling = pooling
        self.use_norm = use_norm
        self.use_residual = use_residual
        self.patch_embed = patch_embed

        if patch_embed == "linear":
            self.hilbert_scan = HilbertScan(image_size=64, patch_size=patch_size)
            patch_dim = self.hilbert_channels * patch_size * patch_size
            self.uproject = nn.Linear(patch_dim, d_model)
            self.conv_stem = None
        elif patch_embed == "conv":
            self.conv_stem = ConvPatchStem(self.hilbert_channels, d_model, patch_size)
            self.hilbert_scan = HilbertScan(image_size=64 // patch_size, patch_size=1)
            self.uproject = nn.Identity()
        else:
            raise ValueError(f"Unknown patch_embed type {patch_embed}")

        self.s4_layers = nn.ModuleList([
            S4DConv(d_model=d_model, d_state=s4_state, transposed=False)
            for _ in range(num_layers)
        ])
        self.acts = nn.ModuleList([nn.GELU() for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)]) if use_norm else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if pooling == "last":
            self.take_last = TakeLastTimestep()
        elif pooling == "mean":
            self.take_last = None
        else:
            raise ValueError(f"Unknown pooling type {pooling}")

        self.fc = nn.Linear(d_model, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_logits=True):
        if self.patch_embed == "conv":
            feat = self.conv_stem(x)
            x_seq = self.hilbert_scan(feat)
            h = self.uproject(x_seq)
        else:
            x_seq = self.hilbert_scan(x)
            h = self.uproject(x_seq)

        for i, (s4_layer, act) in enumerate(zip(self.s4_layers, self.acts)):
            residual = h
            h_in = self.norms[i](h) if self.use_norm else h
            h_out, _ = s4_layer(h_in)
            h_out = act(h_out)
            h_out = self.drop(h_out)
            h = residual + h_out if self.use_residual else h_out

        pooled = h.mean(dim=1) if self.take_last is None else self.take_last(h)
        logits = self.fc(pooled)

        if return_logits:
            return logits
        return self.softmax(logits)


# Alias used by report-discrepency-testing.ipynb's controlled recipe-crossing
# study -- same class, diffed byte-identical to the definition above.
MainStudyGalaxyClassifier = GalaxyClassifierS4DFast
