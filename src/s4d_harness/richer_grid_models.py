"""
Richer-stem family: the stem-depth x S4D-layer-count grid (Table richer-grid /
Table followup-grid in the LaTeX report).

Two implementations live here, both extracted from real notebook code, kept
separate and clearly labeled because they are genuinely two different
(closely related) codebases that produced two different tables:

1. ``Stem1DetailOnly`` ... ``Stem4Full`` + ``GalaxyClassifierGrid``
   The ORIGINAL grid architecture, extracted verbatim from
   scripts/legacy/train_ablation.py (itself %%writefile-extracted from
   notebooks/lodhi-training-re-runs(1).ipynb). This produced the
   short-recipe (40-epoch, seed 42) numbers in the report's Table
   richer-grid / kaggle-full-grid.

2. ``RicherStem`` + ``RicherGridModel``
   A deliberate RECONSTRUCTION built in notebooks/future-work-training.ipynb
   (and future-work-training-v3.ipynb / s4d-future-work-validation-resumable-
   kaggle.ipynb) to reproduce the same 13 parameter counts under the MAIN
   recipe, for the report's Section "Follow-Up Validation" / Table
   followup-grid. The notebook's own comment on this class states it is a
   reconstruction chosen to match reported parameter counts, not a byte-for-
   byte copy of (1) -- both are kept here rather than silently merged, since
   conflating them would misrepresent which run produced which table.

Both depend on model.hilbert.HilbertScan, model.tlts.TakeLastTimestep, and
model.s4d_recurrent.S4D from the shared `model/` package at the repo root.
"""
import torch
import torch.nn as nn

from model.hilbert import HilbertScan
from model.tlts import TakeLastTimestep
from model.s4d_recurrent import S4D

# ============================================================================
# 1. ORIGINAL grid (train_ablation.py / lodhi-training-re-runs, short recipe)
# ============================================================================

D_MODEL = 64
D_STATE = 64
MID_CHANNELS = 32
STEM_DROP = 0.1
HEAD_DROP = 0.2


def _gn(channels):
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


class Stem4Full(nn.Module):
    """4 conv layers, matches model/cnn_stem.py (reduction=16). Grid: 16x16."""
    NAME, N_LAYERS = "stem_full (4L)", 4

    def __init__(self, in_channels):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, MID_CHANNELS, 3, 1, 1)
        self.stem_norm = _gn(MID_CHANNELS)
        self.stem_act = nn.GELU()
        self.res_conv = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, 3, 1, 1)
        self.res_norm = _gn(MID_CHANNELS)
        self.res_act = nn.GELU()
        self.drop = nn.Dropout2d(STEM_DROP)
        self.down1 = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, 3, 2, 1)
        self.down1_norm = _gn(MID_CHANNELS)
        self.down1_act = nn.GELU()
        self.down2 = nn.Conv2d(MID_CHANNELS, D_MODEL, 3, 2, 1)
        self.down2_norm = _gn(D_MODEL)
        self.down2_act = nn.GELU()

    def forward(self, x):
        x = self.stem_act(self.stem_norm(self.stem_conv(x)))
        r = self.res_act(self.res_norm(self.res_conv(x)))
        x = self.drop(x + r)
        x = self.down1_act(self.down1_norm(self.down1(x)))
        x = self.down2_act(self.down2_norm(self.down2(x)))
        return x  # (B, D_MODEL, 16, 16)


class Stem3NoResidual(nn.Module):
    """3 conv layers -- drop the residual refinement block. Grid: 16x16."""
    NAME, N_LAYERS = "stem_3 (drop res_conv)", 3

    def __init__(self, in_channels):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, MID_CHANNELS, 3, 1, 1)
        self.stem_norm = _gn(MID_CHANNELS)
        self.stem_act = nn.GELU()
        self.drop = nn.Dropout2d(STEM_DROP)
        self.down1 = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, 3, 2, 1)
        self.down1_norm = _gn(MID_CHANNELS)
        self.down1_act = nn.GELU()
        self.down2 = nn.Conv2d(MID_CHANNELS, D_MODEL, 3, 2, 1)
        self.down2_norm = _gn(D_MODEL)
        self.down2_act = nn.GELU()

    def forward(self, x):
        x = self.stem_act(self.stem_norm(self.stem_conv(x)))
        x = self.drop(x)
        x = self.down1_act(self.down1_norm(self.down1(x)))
        x = self.down2_act(self.down2_norm(self.down2(x)))
        return x  # (B, D_MODEL, 16, 16)


class Stem2SingleDownsample(nn.Module):
    """2 conv layers -- one detail conv + one downsample straight to d_model.
    Grid: 32x32 (longer S4D sequence: 1024 vs 256 for the 16x16 variants)."""
    NAME, N_LAYERS = "stem_2 (1 detail + 1 down)", 2

    def __init__(self, in_channels):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, MID_CHANNELS, 3, 1, 1)
        self.stem_norm = _gn(MID_CHANNELS)
        self.stem_act = nn.GELU()
        self.drop = nn.Dropout2d(STEM_DROP)
        self.down = nn.Conv2d(MID_CHANNELS, D_MODEL, 3, 2, 1)
        self.down_norm = _gn(D_MODEL)
        self.down_act = nn.GELU()

    def forward(self, x):
        x = self.stem_act(self.stem_norm(self.stem_conv(x)))
        x = self.drop(x)
        x = self.down_act(self.down_norm(self.down(x)))
        return x  # (B, D_MODEL, 32, 32)


class Stem1DetailOnly(nn.Module):
    """1 conv layer -- detail extraction only, no downsampling.
    Grid: 64x64 (full-resolution S4D sequence: 4096, most expensive)."""
    NAME, N_LAYERS = "stem_1 (detail only)", 1

    def __init__(self, in_channels):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, D_MODEL, 3, 1, 1)
        self.stem_norm = _gn(D_MODEL)
        self.stem_act = nn.GELU()
        self.drop = nn.Dropout2d(STEM_DROP)

    def forward(self, x):
        x = self.stem_act(self.stem_norm(self.stem_conv(x)))
        x = self.drop(x)
        return x  # (B, D_MODEL, 64, 64)


STEM_LADDER = [Stem4Full, Stem3NoResidual, Stem2SingleDownsample, Stem1DetailOnly]
S4_LAYER_COUNTS = [0, 1, 2]


class GalaxyClassifierGrid(nn.Module):
    """CNN stem (any depth from STEM_LADDER) + N stacked S4D layers, N in {0,1,2}.

    N=0 -> pure CNN: stem -> global-avg-pool -> fc (no S4D at all)
    N>0 -> stem -> Hilbert-scan -> N x (S4D + GELU) -> take-last -> fc
    """

    def __init__(self, stem_cls, num_s4_layers, in_channels):
        super().__init__()
        assert num_s4_layers in (0, 1, 2)
        self.stem_name = stem_cls.NAME
        self.stem_layers = stem_cls.N_LAYERS
        self.num_s4_layers = num_s4_layers

        self.stem = stem_cls(in_channels)

        if num_s4_layers > 0:
            self.s4_layers = nn.ModuleList([
                S4D(d_model=D_MODEL, d_state=D_STATE, transposed=False)
                for _ in range(num_s4_layers)
            ])
            self.acts = nn.ModuleList([nn.GELU() for _ in range(num_s4_layers)])
            self.take_last = TakeLastTimestep()
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head_drop = nn.Dropout(HEAD_DROP)
        self.fc = nn.Linear(D_MODEL, 4)
        self.softmax = nn.Softmax(dim=-1)

        self._hilbert = None  # built lazily once we know the stem's grid size
        self._hilbert_n = None

    def forward(self, x, return_logits=False):
        feat = self.stem(x)  # (B, D_MODEL, grid, grid)

        if self.num_s4_layers == 0:
            out = self.global_pool(feat).flatten(1)  # (B, D_MODEL)
        else:
            grid = feat.shape[-1]
            if self._hilbert is None or self._hilbert_n != grid:
                self._hilbert = HilbertScan(n=grid).to(feat.device)
                self._hilbert_n = grid
            seq = self._hilbert(feat)  # (B, seq_len, D_MODEL)
            for s4, act in zip(self.s4_layers, self.acts):
                seq, _ = s4(seq)
                seq = act(seq)
            out = self.take_last(seq)  # (B, D_MODEL)

        out = self.head_drop(out)
        logits = self.fc(out)
        return logits if return_logits else self.softmax(logits)


# Original short-recipe grid: exact reported parameter counts, for tests.
ORIGINAL_GRID_REPORT_PARAMS = {
    (4, 0): 38468, (4, 1): 46788, (4, 2): 55108,
    (3, 0): 29156, (3, 1): 37476, (3, 2): 45796,
    (2, 0): 19844, (2, 1): 28164, (2, 2): 36484,
    (1, 0): 2180, (1, 1): 10500, (1, 2): 18820,
}


def make_original_grid_model(stem_depth, num_s4_layers, in_channels=3):
    """stem_depth in {1,2,3,4} maps to Stem1DetailOnly..Stem4Full."""
    stem_cls = {4: Stem4Full, 3: Stem3NoResidual, 2: Stem2SingleDownsample, 1: Stem1DetailOnly}[stem_depth]
    return GalaxyClassifierGrid(stem_cls, num_s4_layers, in_channels)


# ============================================================================
# 2. RECONSTRUCTED grid (future-work-training.ipynb, main recipe follow-up)
# ============================================================================

class RicherStem(nn.Module):
    """Reconstruction of the report's 1/2/3/4-layer richer stems, built to
    reproduce the report's published parameter counts:
      depth 1 -> 2,180 params incl. classifier
      depth 2 -> 19,844
      depth 3 -> 29,156
      depth 4 -> 38,468

    All convolution outputs are GroupNorm + GELU. The 4-layer variant includes
    the documented full-resolution residual block. Spatial side length is 64
    for depth 1 and 16 for depths 2-4.
    """

    def __init__(self, depth, in_channels=3, d_model=64, mid_channels=32, dropout=0.1):
        super().__init__()
        if depth not in (1, 2, 3, 4):
            raise ValueError(depth)
        self.depth = depth
        self.out_channels = d_model
        self.grid = 64 if depth == 1 else 16
        self.dropout = nn.Identity()

        def gn(ch):
            groups = 8 if ch % 8 == 0 else 1
            return nn.GroupNorm(groups, ch)

        if depth == 1:
            self.conv1 = nn.Conv2d(in_channels, d_model, 3, stride=1, padding=1)
            self.norm1 = gn(d_model)
            self.act1 = nn.GELU()
        elif depth == 2:
            self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride=1, padding=1)
            self.norm1 = gn(mid_channels)
            self.act1 = nn.GELU()
            self.conv2 = nn.Conv2d(mid_channels, d_model, 3, stride=4, padding=1)
            self.norm2 = gn(d_model)
            self.act2 = nn.GELU()
        elif depth == 3:
            self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride=1, padding=1)
            self.norm1 = gn(mid_channels)
            self.act1 = nn.GELU()
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1)
            self.norm2 = gn(mid_channels)
            self.act2 = nn.GELU()
            self.conv3 = nn.Conv2d(mid_channels, d_model, 3, stride=2, padding=1)
            self.norm3 = gn(d_model)
            self.act3 = nn.GELU()
        else:
            # Matches the CNNStem already used elsewhere: full-res detail
            # conv + full-res residual conv + two stride-2 downsamples.
            self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride=1, padding=1)
            self.norm1 = gn(mid_channels)
            self.act1 = nn.GELU()
            self.res_conv = nn.Conv2d(mid_channels, mid_channels, 3, stride=1, padding=1)
            self.res_norm = gn(mid_channels)
            self.res_act = nn.GELU()
            self.drop = nn.Dropout2d(dropout)
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=2, padding=1)
            self.norm2 = gn(mid_channels)
            self.act2 = nn.GELU()
            self.conv3 = nn.Conv2d(mid_channels, d_model, 3, stride=2, padding=1)
            self.norm3 = gn(d_model)
            self.act3 = nn.GELU()

    def forward(self, x):
        if self.depth == 1:
            return self.act1(self.norm1(self.conv1(x)))
        if self.depth == 2:
            x = self.act1(self.norm1(self.conv1(x)))
            return self.act2(self.norm2(self.conv2(x)))
        if self.depth == 3:
            x = self.act1(self.norm1(self.conv1(x)))
            x = self.act2(self.norm2(self.conv2(x)))
            return self.act3(self.norm3(self.conv3(x)))
        x = self.act1(self.norm1(self.conv1(x)))
        r = self.res_act(self.res_norm(self.res_conv(x)))
        x = self.drop(x + r)
        x = self.act2(self.norm2(self.conv2(x)))
        return self.act3(self.norm3(self.conv3(x)))


class RicherGridModel(nn.Module):
    def __init__(self, stem_depth, num_s4_layers, d_model=64, s4_state=64, num_classes=4, stem_dropout=0.1):
        super().__init__()
        if num_s4_layers not in (0, 1, 2):
            raise ValueError(num_s4_layers)
        self.stem_depth = stem_depth
        self.num_s4_layers = num_s4_layers
        self.cnn_stem = RicherStem(stem_depth, d_model=d_model, dropout=stem_dropout)
        self.grid = self.cnn_stem.grid
        self.seq_len = self.grid * self.grid
        self.hilbert_scan = HilbertScan(n=self.grid)
        self.s4_layers = nn.ModuleList([
            S4D(d_model=d_model, d_state=s4_state, transposed=False)
            for _ in range(num_s4_layers)
        ])
        self.acts = nn.ModuleList([nn.GELU() for _ in range(num_s4_layers)])
        self.take_last = TakeLastTimestep()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, return_logits=True):
        feat = self.cnn_stem(x)
        h = self.hilbert_scan(feat)
        for layer, act in zip(self.s4_layers, self.acts):
            h, _ = layer(h)
            h = act(h)
        pooled = self.take_last(h)
        logits = self.fc(pooled)
        return logits if return_logits else torch.softmax(logits, dim=-1)


def make_richer_grid_model(stem_depth, num_s4_layers):
    return RicherGridModel(stem_depth, num_s4_layers)


# Reconstructed grid: same reported parameter counts as the original grid
# (that consistency is itself part of what report Section "Follow-Up
# Validation" checked before trusting the reconstruction's main-recipe runs).
RICHER_REPORT_PARAMS = {
    (1, 0): 2180, (1, 1): 10500, (1, 2): 18820,
    (2, 0): 19844, (2, 1): 28164, (2, 2): 36484,
    (3, 0): 29156, (3, 1): 37476, (3, 2): 45796,
    (4, 0): 38468, (4, 1): 46788, (4, 2): 55108,
}
RICHER_REPORT_SEQ = {1: 4096, 2: 256, 3: 256, 4: 256}
