import torch
import torch.nn as nn


class CNNStem(nn.Module):
    """
    Convolutional stem for the CNN-stem -> S4D hybrid classifier.

    RESEARCH VERSION (post-course). The course version of this stem was
    constrained to plain Conv+GELU (no BatchNorm/LayerNorm) for eventual
    bare-metal RISC-V portability. That constraint is dropped here since
    we're now optimizing purely for accuracy -- if a bare-metal export is
    ever needed again, GroupNorm's running stats can be folded into the
    preceding conv's weights at export time (standard conv-BN/GN fusion),
    so this doesn't have to be a permanent trade-off even for that goal.

    Key differences from the course version:
      1. A stride-1 "detail" conv runs FIRST, at full input resolution,
         before any downsampling happens. The old stem's first conv was
         already stride-2, so it only ever saw a raw 3x3 window of
         un-processed pixels before halving resolution -- thin, low-
         contrast structures (e.g. dust lanes in edge-on spirals, which
         is exactly the signal needed to separate Smooth Cigar from
         Edge-on Disk) had no chance to be extracted before being pooled
         away. Now there's a full-res feature-extraction pass first.
      2. GroupNorm after every conv (stable training, no batch-size
         dependence, no running stats to worry about -- unlike
         BatchNorm, GroupNorm's stats are computed per-sample so it
         behaves identically in train/eval).
      3. More channel capacity (mid_channels default raised 16 -> 32).
      4. Residual add on the stride-1 block (cheap, helps optimization,
         doesn't change spatial dims so it's a free add).

    Two variants, selected via `reduction` (same semantics as before):
      - reduction=16: three conv stages, 64x64 -> 64x64 (stride1) ->
        32x32 -> 16x16   => 16x sequence-length cut (4096->256)
      - reduction=4:  64x64 -> 64x64 (stride1) -> 32x32
                                                    => 4x cut (4096->1024)

    Parameters
    ----------
    in_channels : int
        Number of input image channels (1 grayscale, 3 RGB). Use 3 --
        color carries the dust-lane / reddening signal that grayscale
        (channel-averaged) input throws away.
    d_model : int, optional
        Output channel count of the stem, feeding S4D's d_model. Default 64.
    mid_channels : int, optional
        Hidden channel width of the stem's early conv stages. Default 32
        (was 16 in the course version -- more capacity now that accuracy,
        not param-count / embedded footprint, is the objective).
    reduction : int, optional
        Spatial / sequence-length reduction factor. One of {4, 16}.
        Default 16.
    dropout : float, optional
        Spatial dropout (Dropout2d) applied after the stride-1 block, as
        light regularization for the ~8k-image dataset. Default 0.1.

    Input
    -----
    x : torch.Tensor, shape (B, in_channels, 64, 64)

    Returns
    -------
    torch.Tensor
        shape (B, d_model, 16, 16) if reduction=16,
        shape (B, d_model, 32, 32) if reduction=4.
    """

    def __init__(self, in_channels, d_model=64, mid_channels=32, reduction=16, dropout=0.1):
        super().__init__()
        if reduction not in (4, 16):
            raise ValueError(f"reduction must be 4 or 16, got {reduction}")
        self.reduction = reduction

        def gn(channels):
            # GroupNorm needs num_groups | channels; 8 groups is a safe
            # default for the channel counts used here (32, 64).
            groups = 8 if channels % 8 == 0 else 1
            return nn.GroupNorm(groups, channels)

        # --- Stage 0: full-resolution detail extraction (stride 1) ---
        # This is the change that matters most: features are computed at
        # the input's native 64x64 resolution before anything is thrown
        # away, so thin/low-contrast structures (dust lanes, arm edges)
        # actually get a chance to be represented.
        self.stem_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.stem_norm = gn(mid_channels)
        self.stem_act = nn.GELU()

        self.res_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.res_norm = gn(mid_channels)
        self.res_act = nn.GELU()
        self.drop = nn.Dropout2d(dropout)

        # --- Downsampling stages ---
        if reduction == 16:
            # 64 -> 32 -> 16
            self.down1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1)
            self.down1_norm = gn(mid_channels)
            self.down1_act = nn.GELU()

            self.down2 = nn.Conv2d(mid_channels, d_model, kernel_size=3, stride=2, padding=1)
            self.down2_norm = gn(d_model)
            self.down2_act = nn.GELU()
        else:
            # 64 -> 32
            self.down1 = nn.Conv2d(mid_channels, d_model, kernel_size=3, stride=2, padding=1)
            self.down1_norm = gn(d_model)
            self.down1_act = nn.GELU()
            self.down2 = None

    def forward(self, x):
        # x: (B, in_channels, 64, 64)
        x = self.stem_act(self.stem_norm(self.stem_conv(x)))       # full-res feature extraction
        r = self.res_act(self.res_norm(self.res_conv(x)))
        x = x + r                                                   # residual, still full-res
        x = self.drop(x)

        x = self.down1_act(self.down1_norm(self.down1(x)))
        if self.down2 is not None:
            x = self.down2_act(self.down2_norm(self.down2(x)))
        return x
