import torch
import torch.nn as nn

from .cnn_stem import CNNStem


class GalaxyClassifierCNNOnly(nn.Module):
    """
    CNN-only baseline: same CNNStem as GalaxyClassifierCNNS4D, but with NO
    S4D layers at all. Feature map is pooled directly (global average
    pooling) and classified.

    Why this model exists
    ----------------------
    Every result so far has tested "CNN stem -> S4D". Nothing so far has
    tested "CNN alone" at a matched parameter budget. That leaves an open
    question raised directly by the TA:

      1. Would a ~43K-param CNN alone reach similar accuracy to the
         ~55K-param CNN+S4D hybrid (86.80%)?
      2. Would a ~60K-param CNN alone reach similar accuracy to the
         ~63K-param CNN+S4D 3-layer hybrid (86.65%)?
      3. Is S4D actually adding signal beyond what the CNN stem alone
         already provides, or is the CNN stem doing all the work (echoing
         the color ablation finding, where the "obvious" explanation
         wasn't the real one)?
      4. How small can the CNN get before accuracy collapses -- i.e.
         where is the actual capacity floor for this task?

    This class, together with scripts/train_cnn_only.py, runs three sizes
    (~10K, ~43K, ~60K params) to answer all four questions with real
    numbers instead of assumptions on either side.

    Design choice: global average pooling over the stem's output grid is
    used as the CNN-only readout, because it is the direct analog of the
    S4D hybrid's mean-pooling over the Hilbert-scanned sequence -- same
    "average all spatial/sequence positions" idea, just without S4D's
    sequential processing in between. This keeps the comparison about
    S4D specifically, not about the readout strategy also changing.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes (default 4).
    colored : bool, optional
        RGB (3-channel) if True, grayscale (1-channel) if False. Default True.
    stem_reduction : int, optional
        Passed through to CNNStem, one of {4, 16}. Default 16 (matches the
        winning hybrid configuration).
    mid_channels : int, optional
        Stem hidden channel width. Default 32.
    d_model : int, optional
        Stem output channel width (and refine-conv width, if used). Default 64.
    stem_dropout : float, optional
        Dropout2d inside the stem. Default 0.1.
    head_dropout : float, optional
        Dropout applied to the pooled vector before the classifier head. Default 0.2.
    use_refine_conv : bool, optional
        If True, adds one extra 1x1 conv + GroupNorm + GELU after the stem,
        before pooling -- used to hit the ~43K/~60K parameter targets
        without changing the stem's own architecture. Default True.
    """

    def __init__(self, num_classes=4, colored=True, stem_reduction=16,
                 mid_channels=32, d_model=64, stem_dropout=0.1,
                 head_dropout=0.2, use_refine_conv=True):
        super().__init__()
        if stem_reduction not in (4, 16):
            raise ValueError(f"stem_reduction must be 4 or 16, got {stem_reduction}")

        self.hilbert_channels = 1 if not colored else 3
        self.d_model = d_model
        self.stem_reduction = stem_reduction
        self.use_refine_conv = use_refine_conv

        grid = 64 // (4 if stem_reduction == 16 else 2)
        self.seq_len = grid * grid  # kept for API/logging parity with the hybrid model

        self.cnn_stem = CNNStem(
            in_channels=self.hilbert_channels,
            d_model=d_model,
            mid_channels=mid_channels,
            reduction=stem_reduction,
            dropout=stem_dropout,
        )

        if use_refine_conv:
            # 1x1 conv: adds a small amount of extra depth/capacity so the
            # small/large variants can be tuned to specific parameter
            # budgets (~43K, ~60K) without changing the stem's own
            # architecture (which is the part already validated by the
            # earlier ablation).
            self.refine_conv = nn.Conv2d(d_model, d_model, kernel_size=1)
            groups = 8 if d_model % 8 == 0 else 1
            self.refine_norm = nn.GroupNorm(groups, d_model)
            self.refine_act = nn.GELU()
        else:
            self.refine_conv = None

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (B, d_model, H, W) -> (B, d_model, 1, 1)
        self.head_drop = nn.Dropout(head_dropout)
        self.fc = nn.Linear(d_model, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_logits=False):
        # x: (B, hilbert_channels, 64, 64)
        feat = self.cnn_stem(x)  # (B, d_model, grid, grid)

        if self.refine_conv is not None:
            feat = self.refine_act(self.refine_norm(self.refine_conv(feat)))

        pooled = self.global_pool(feat).flatten(1)  # (B, d_model)
        pooled = self.head_drop(pooled)

        logits = self.fc(pooled)  # (B, num_classes)

        if return_logits:
            return logits
        return self.softmax(logits)
