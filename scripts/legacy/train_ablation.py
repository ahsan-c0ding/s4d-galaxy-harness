#!/usr/bin/env python
"""
Full Grid Ablation — CNN stem depth × S4D layer count, every combination,
compared against the raw S4D-only baseline.

WHAT THIS ANSWERS
------------------
Two ablations that existed separately in this project are combined here
into one full grid, run under one identical training regime so every
number in the output is directly comparable:

  1. "How few CNN stem layers can we get away with?"
     (previously: a 1D ladder of 4 -> 1 stem layers, S4D depth held fixed)
  2. "How many S4D layers actually help on top of the CNN stem?"
     (previously: a 1D ladder of 0 -> 2 S4D layers, stem held fixed)

Instead of two separate 1D sweeps, this script trains the full 2D grid:

    stem depth  ∈ {1, 2, 3, 4}   (Stem1DetailOnly ... Stem4Full)
    S4D layers  ∈ {0, 1, 2}

  = 12 CNN-stem models
  + 1 raw-pixel S4D-only baseline (no CNN stem at all, 2 fixed S4D layers,
    the existing model/gclassifier.py model)
  ---------------------------------------------------------------------
  = 13 models total, every stem-depth × S4D-depth combination that makes
    sense for this architecture family.

  (CNN-only is not trained as a separate model class: it's exactly the
  S4D-layers=0 column of the grid, for each of the 4 stem depths, so it
  falls out of the grid for free instead of being duplicated.)

GOAL, per your instructions
----------------------------
Not "what's the single most accurate model" — the goal is to map the
whole accuracy / param-count / memory trade-off surface, so you can pick
the smallest model that stays within an acceptable accuracy drop. Every
model reports both a size cost (parameter count, always available) and a
runtime memory cost (peak CUDA memory during a forward pass, when a GPU
is available) since parameter count alone doesn't capture activation
memory, which is what actually matters for the "low memory" goal on the
CNN-heavy variants (bigger feature maps = more activation memory even
at a fixed param count).

METRICS COLLECTED PER MODEL
----------------------------
  - Test accuracy
  - Confusion matrix (plotted)
  - Precision / Recall / F1 — per class, macro, and weighted
  - ROC-AUC — per class (one-vs-rest), macro, weighted (ROC curves plotted)
  - Parameter count (total, and stem-only where applicable)
  - Peak CUDA memory for a forward pass (GPU only)
  - Per-sample inference latency (median, ms)
  - Full training-loss and validation-accuracy curves (plotted)
  - Robustness to input noise: test accuracy under additive Gaussian pixel
    noise at several sigma levels (plotted per model and combined)

ASSUMPTION FLAGGED EXPLICITLY: "noise" wasn't defined in the request, so
it's interpreted here as an input-robustness sweep (accuracy vs Gaussian
noise sigma added to the test images at eval time). If you meant something
else (e.g. label noise during training, or a specific corruption type),
tell me and I'll swap the sweep out — the hook is isolated in
`noise_robustness_sweep()` below.

CRITICAL: every model uses the SAME seed / LR / epochs / batch size /
optimizer / scheduler / label smoothing / color input / train-val-test
split as the earlier ablation scripts in this project, so all these
numbers are directly comparable to the ones already produced by
scripts/train_ablation.py.

Run from the repo root:
    python scripts/train_full_ablation.py
"""

import importlib.util
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ── repo root on path ────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import GalaxyClassifierS4D
from model.cnn_stem import CNNStem
from model.hilbert import HilbertScan
from model.tlts import TakeLastTimestep
from model.s4d_recurrent import S4D
from model.functions import load_data

import kaggle_extras
KAGGLE_OUT_DIR = kaggle_extras.output_dir("train_ablation")

# ══════════════════════════════════════════════════════════════════════════
#  TRAINING CONFIG — kept identical to scripts/train_ablation.py on purpose
# ══════════════════════════════════════════════════════════════════════════
RNG_SEED        = 42
BATCH_SIZE      = 64
LR              = 0.001
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.05
EPOCHS          = 40
COLORED         = True
CLASS_NAMES     = ["Smooth Round", "Smooth Cigar", "Edge-on Disk", "Unbarred Spiral"]
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

D_MODEL      = 64
D_STATE      = 64
MID_CHANNELS = 32
STEM_DROP    = 0.1
HEAD_DROP    = 0.2

# Noise-robustness sweep levels (std-dev of additive Gaussian pixel noise,
# images are normalized to [0,1] before noise is added)
NOISE_SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3]

# "Acceptable" accuracy drop for the final low-memory recommendation
ACCEPTABLE_DROP_PP = 1.5

OUT_DIR = os.path.join(_REPO_ROOT, "ablation_results", "full_grid")
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed: int = RNG_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════
#  CNN STEM LADDER — 4 depths, ladder from 4 layers down to 1
#  (same variants used in the earlier stem-depth ablation, kept verbatim so
#  those numbers stay reproducible from this script too)
# ══════════════════════════════════════════════════════════════════════════
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
        self.stem_act  = nn.GELU()
        self.res_conv  = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, 3, 1, 1)
        self.res_norm  = _gn(MID_CHANNELS)
        self.res_act   = nn.GELU()
        self.drop      = nn.Dropout2d(STEM_DROP)
        self.down1      = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, 3, 2, 1)
        self.down1_norm = _gn(MID_CHANNELS)
        self.down1_act  = nn.GELU()
        self.down2      = nn.Conv2d(MID_CHANNELS, D_MODEL, 3, 2, 1)
        self.down2_norm = _gn(D_MODEL)
        self.down2_act  = nn.GELU()

    def forward(self, x):
        x = self.stem_act(self.stem_norm(self.stem_conv(x)))
        r = self.res_act(self.res_norm(self.res_conv(x)))
        x = self.drop(x + r)
        x = self.down1_act(self.down1_norm(self.down1(x)))
        x = self.down2_act(self.down2_norm(self.down2(x)))
        return x  # (B, D_MODEL, 16, 16)


class Stem3NoResidual(nn.Module):
    """3 conv layers — drop the residual refinement block. Grid: 16x16."""
    NAME, N_LAYERS = "stem_3 (drop res_conv)", 3

    def __init__(self, in_channels):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, MID_CHANNELS, 3, 1, 1)
        self.stem_norm = _gn(MID_CHANNELS)
        self.stem_act  = nn.GELU()
        self.drop      = nn.Dropout2d(STEM_DROP)
        self.down1      = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, 3, 2, 1)
        self.down1_norm = _gn(MID_CHANNELS)
        self.down1_act  = nn.GELU()
        self.down2      = nn.Conv2d(MID_CHANNELS, D_MODEL, 3, 2, 1)
        self.down2_norm = _gn(D_MODEL)
        self.down2_act  = nn.GELU()

    def forward(self, x):
        x = self.stem_act(self.stem_norm(self.stem_conv(x)))
        x = self.drop(x)
        x = self.down1_act(self.down1_norm(self.down1(x)))
        x = self.down2_act(self.down2_norm(self.down2(x)))
        return x  # (B, D_MODEL, 16, 16)


class Stem2SingleDownsample(nn.Module):
    """2 conv layers — one detail conv + one downsample straight to d_model.
    Grid: 32x32 (longer S4D sequence: 1024 vs 256 for the 16x16 variants)."""
    NAME, N_LAYERS = "stem_2 (1 detail + 1 down)", 2

    def __init__(self, in_channels):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, MID_CHANNELS, 3, 1, 1)
        self.stem_norm = _gn(MID_CHANNELS)
        self.stem_act  = nn.GELU()
        self.drop      = nn.Dropout2d(STEM_DROP)
        self.down      = nn.Conv2d(MID_CHANNELS, D_MODEL, 3, 2, 1)
        self.down_norm = _gn(D_MODEL)
        self.down_act  = nn.GELU()

    def forward(self, x):
        x = self.stem_act(self.stem_norm(self.stem_conv(x)))
        x = self.drop(x)
        x = self.down_act(self.down_norm(self.down(x)))
        return x  # (B, D_MODEL, 32, 32)


class Stem1DetailOnly(nn.Module):
    """1 conv layer — detail extraction only, no downsampling.
    Grid: 64x64 (full-resolution S4D sequence: 4096, most expensive)."""
    NAME, N_LAYERS = "stem_1 (detail only)", 1

    def __init__(self, in_channels):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, D_MODEL, 3, 1, 1)
        self.stem_norm = _gn(D_MODEL)
        self.stem_act  = nn.GELU()
        self.drop      = nn.Dropout2d(STEM_DROP)

    def forward(self, x):
        x = self.stem_act(self.stem_norm(self.stem_conv(x)))
        x = self.drop(x)
        return x  # (B, D_MODEL, 64, 64)


STEM_LADDER = [Stem4Full, Stem3NoResidual, Stem2SingleDownsample, Stem1DetailOnly]
S4_LAYER_COUNTS = [0, 1, 2]


# ══════════════════════════════════════════════════════════════════════════
#  GRID MODEL — any stem depth × any S4D layer count (0, 1, or 2)
# ══════════════════════════════════════════════════════════════════════════
class GalaxyClassifierGrid(nn.Module):
    """CNN stem (any depth from STEM_LADDER) + N stacked S4D layers, N in {0,1,2}.

    N=0 -> pure CNN: stem -> global-avg-pool -> fc (no S4D at all)
    N>0 -> stem -> Hilbert-scan -> N x (S4D + GELU) -> take-last -> fc

    Everything downstream of the stem is identical in shape/logic to the
    rest of this project's hybrid model, so accuracy differences trace
    back only to (stem depth, S4D depth), not to some other change riding
    along with it.
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

        self._hilbert = None   # built lazily once we know the stem's grid size
        self._hilbert_n = None

    def forward(self, x, return_logits=False):
        feat = self.stem(x)               # (B, D_MODEL, grid, grid)

        if self.num_s4_layers == 0:
            out = self.global_pool(feat).flatten(1)   # (B, D_MODEL)
        else:
            grid = feat.shape[-1]
            if self._hilbert is None or self._hilbert_n != grid:
                self._hilbert = HilbertScan(n=grid).to(feat.device)
                self._hilbert_n = grid
            seq = self._hilbert(feat)                  # (B, seq_len, D_MODEL)
            for s4, act in zip(self.s4_layers, self.acts):
                seq, _ = s4(seq)
                seq = act(seq)
            out = self.take_last(seq)                  # (B, D_MODEL)

        out = self.head_drop(out)
        logits = self.fc(out)
        return logits if return_logits else self.softmax(logits)


# ══════════════════════════════════════════════════════════════════════════
#  DATA — loaded once, identical split for every model
# ══════════════════════════════════════════════════════════════════════════
def load_and_split():
    print("Loading data …")
    set_seed()

    X_train_all, y_onehot_train, y_train_all = load_data(
        root="./data", download=True, train=True, colored=COLORED
    )
    X_test, y_onehot_test, y_test = load_data(
        root="./data", download=True, train=False, colored=COLORED
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_onehot_train,
        test_size=0.2, random_state=RNG_SEED, stratify=y_train_all
    )

    def to_loader(X, y, shuffle):
        return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)
    test_loader  = to_loader(X_test,  y_onehot_test, shuffle=False)

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return train_loader, val_loader, test_loader, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════
#  NOISE ROBUSTNESS SWEEP
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def noise_robustness_sweep(model, X_test, y_test, sigmas=NOISE_SIGMAS):
    """Test accuracy under additive Gaussian pixel noise at each sigma.
    Images are assumed normalized to [0, 1]; noised inputs are clamped
    back into [0, 1] before the forward pass, matching what the model
    was trained to expect."""
    model.eval()
    accs = []
    X_test_t = torch.as_tensor(X_test, dtype=torch.float32)
    y_test_t = torch.as_tensor(y_test)
    loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE)
    for sigma in sigmas:
        correct, total = 0, 0
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            if sigma > 0:
                xb = torch.clamp(xb + torch.randn_like(xb) * sigma, 0.0, 1.0)
            logits = model(xb, return_logits=True)
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        accs.append(correct / total)
    return accs  # list, same order as sigmas


# ══════════════════════════════════════════════════════════════════════════
#  TRAIN / EVAL ONE MODEL — full metric suite
# ══════════════════════════════════════════════════════════════════════════
def run_model(name, model, train_loader, val_loader, test_loader, X_test, y_test,
              extra_info=None):
    set_seed()
    model = model.to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    stem_params = sum(p.numel() for p in model.stem.parameters()) if hasattr(model, "stem") else None

    print(f"\n{'='*60}\n  Model : {name}\n  Params: {param_count:,}" +
          (f"  (stem: {stem_params:,})" if stem_params is not None else "") +
          f"\n  Device: {DEVICE}\n{'='*60}")

    kaggle_extras.print_and_save_summary(
        model, (1, 3 if COLORED else 1, 64, 64), name, KAGGLE_OUT_DIR)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    train_losses, val_accs = [], []
    t_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            y_int = y_batch.argmax(dim=1) if y_batch.dim() == 2 else y_batch

            optimizer.zero_grad()
            logits = model(X_batch, return_logits=True)
            loss = criterion(logits, y_int)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_int = y_batch.argmax(dim=1) if y_batch.dim() == 2 else y_batch
                logits = model(X_batch, return_logits=True)
                correct += (logits.argmax(dim=1) == y_int).sum().item()
                total += len(y_int)
        val_accs.append(correct / total)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{EPOCHS}  loss={train_losses[-1]:.4f}  "
                  f"val_acc={val_accs[-1]*100:.2f}%")

    train_time = time.time() - t_start

    # ── test set: predictions + probabilities ───────────────────────────
    model.eval()
    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch, return_logits=True)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = logits.argmax(dim=1).cpu()
            y_int = y_batch.argmax(dim=1) if y_batch.dim() == 2 else y_batch
            all_preds.extend(preds.numpy())
            all_true.extend(y_int.numpy())
            all_probs.extend(probs.numpy())

    test_acc = float(np.mean(np.array(all_preds) == np.array(all_true)))

    # ── precision / recall / f1 ──────────────────────────────────────────
    n_classes = len(CLASS_NAMES)
    labels_arr = list(range(n_classes))
    prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
        all_true, all_preds, labels=labels_arr, average=None, zero_division=0)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_true, all_preds, labels=labels_arr, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        all_true, all_preds, labels=labels_arr, average="weighted", zero_division=0)

    # ── ROC-AUC (one-vs-rest) ────────────────────────────────────────────
    y_true_bin = label_binarize(all_true, classes=labels_arr)
    probs_arr = np.array(all_probs)
    roc_per_class = {}
    for i, cname in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs_arr[:, i])
        roc_per_class[cname] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
    try:
        auc_macro = float(roc_auc_score(y_true_bin, probs_arr, average="macro", multi_class="ovr"))
        auc_weighted = float(roc_auc_score(y_true_bin, probs_arr, average="weighted", multi_class="ovr"))
    except ValueError:
        auc_macro, auc_weighted = float("nan"), float("nan")

    # ── peak memory (GPU only) — the actual runtime memory cost ─────────
    peak_mem_mb = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        sample = torch.tensor(np.asarray(X_test[:BATCH_SIZE]), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            model(sample, return_logits=True)
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # ── inference latency ─────────────────────────────────────────────
    sample = torch.tensor(np.asarray(X_test[:1]), dtype=torch.float32).to(DEVICE)
    timings = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            model(sample, return_logits=True)
            timings.append((time.perf_counter() - t0) * 1000)
    inf_ms = float(np.median(timings))

    # ── noise robustness sweep ──────────────────────────────────────────
    noise_accs = noise_robustness_sweep(model, X_test, y_test)

    weights_path = kaggle_extras.save_weights(model, name, KAGGLE_OUT_DIR)

    print(f"\n  -> Test accuracy : {test_acc*100:.2f}%")
    print(f"  -> Precision(m)  : {prec_macro:.4f}  Recall(m): {rec_macro:.4f}  F1(m): {f1_macro:.4f}  AUC(m): {auc_macro:.4f}")
    print(f"  -> Train time    : {train_time/60:.1f} min   Inference: {inf_ms:.2f} ms/sample")
    if peak_mem_mb is not None:
        print(f"  -> Peak GPU mem  : {peak_mem_mb:.1f} MB (batch={BATCH_SIZE})")
    print(f"  -> Noise sweep   : " + ", ".join(
        f"σ={s}:{a*100:.1f}%" for s, a in zip(NOISE_SIGMAS, noise_accs)))

    result = {
        "name": name,
        "params": param_count,
        "stem_params": stem_params,
        "test_acc": test_acc,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "train_time": train_time,
        "inf_ms": inf_ms,
        "peak_mem_mb": peak_mem_mb,
        "preds": [int(p) for p in all_preds],
        "true": [int(t) for t in all_true],
        "precision_macro": float(prec_macro), "recall_macro": float(rec_macro), "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_w), "recall_weighted": float(rec_w), "f1_weighted": float(f1_w),
        "per_class": {
            cname: {
                "precision": float(prec_c[i]), "recall": float(rec_c[i]), "f1": float(f1_c[i]),
                "support": int(support_c[i]), "roc_auc": roc_per_class[cname]["auc"],
            } for i, cname in enumerate(CLASS_NAMES)
        },
        "roc_auc_macro": auc_macro, "roc_auc_weighted": auc_weighted,
        "roc_curves": roc_per_class,
        "noise_sigmas": NOISE_SIGMAS,
        "noise_accs": noise_accs,
        "weights_path": weights_path,
    }
    if extra_info:
        result.update(extra_info)
    return result


# ══════════════════════════════════════════════════════════════════════════
#  PER-MODEL PLOTS
# ══════════════════════════════════════════════════════════════════════════
def plot_confusion(r, filename):
    cm = confusion_matrix(r["true"], r["preds"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix — {r['name']}\nAcc: {r['test_acc']*100:.2f}%")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()


def plot_roc(r, filename):
    plt.figure(figsize=(6, 5))
    for cname, curve in r["roc_curves"].items():
        plt.plot(curve["fpr"], curve["tpr"], label=f"{cname} (AUC={curve['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.title(f"ROC — {r['name']}\nMacro AUC: {r['roc_auc_macro']:.3f}")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()


def plot_training_curve(r, filename):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(r["train_losses"]); axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].grid(alpha=0.3)
    axes[1].plot(r["val_accs"]); axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].grid(alpha=0.3)
    fig.suptitle(r["name"])
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════════
#  COMBINED / SUMMARY PLOTS
# ══════════════════════════════════════════════════════════════════════════
def plot_accuracy_vs_params(results, filename):
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in results:
        ax.scatter(r["params"] / 1000, r["test_acc"] * 100, s=90)
        ax.annotate(r["name"], (r["params"] / 1000, r["test_acc"] * 100),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel("Total Parameters (thousands)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Full Grid Ablation — Accuracy vs. Parameter Count")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()


def plot_grid_heatmap(results, value_key, title, filename, fmt=".2f", scale=1.0):
    """Heatmap: rows = stem depth (1..4), cols = S4D layer count (0,1,2).
    The raw S4D-only baseline (no stem) isn't part of this 2D grid and is
    reported separately in the summary table/verdict instead."""
    grid_results = [r for r in results if r.get("stem_layers") is not None]
    mat = np.full((4, 3), np.nan)
    for r in grid_results:
        row = 4 - r["stem_layers"]  # stem_layers=4 -> row 0 (top), 1 -> row 3
        col = r["num_s4_layers"]
        mat[row, col] = r[value_key] * scale

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mat, annot=True, fmt=fmt, cmap="viridis",
                xticklabels=["0 S4D", "1 S4D", "2 S4D"],
                yticklabels=["4 layers", "3 layers", "2 layers", "1 layer"],
                ax=ax)
    ax.set_title(title)
    ax.set_xlabel("S4D layer count"); ax.set_ylabel("CNN stem depth")
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()


def plot_metrics_bar(results, filename):
    names = [r["name"] for r in results]
    metrics = {
        "Precision": [r["precision_macro"] for r in results],
        "Recall":    [r["recall_macro"] for r in results],
        "F1":        [r["f1_macro"] for r in results],
        "ROC-AUC":   [r["roc_auc_macro"] for r in results],
    }
    x = np.arange(len(names)); width = 0.2
    fig, ax = plt.subplots(figsize=(16, 5))
    for i, (label, vals) in enumerate(metrics.items()):
        ax.bar(x + (i - 1.5) * width, vals, width, label=label)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score (macro)")
    ax.set_title("Full Grid Ablation — Precision / Recall / F1 / ROC-AUC")
    ax.legend()
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()


def plot_noise_robustness(results, filename):
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in results:
        ax.plot(r["noise_sigmas"], [a * 100 for a in r["noise_accs"]],
                marker="o", label=r["name"])
    ax.set_xlabel("Gaussian noise σ (added to [0,1]-normalized pixels)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Robustness to Input Noise")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()


def plot_summary_bar(results, filename):
    names = [r["name"] for r in results]
    accs = [r["test_acc"] * 100 for r in results]
    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(names, accs, edgecolor="white", width=0.6)
    ax.set_ylim(min(accs) - 5, max(accs) + 3)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Full Grid Ablation — Test Accuracy by Model")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=7)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()


# ══════════════════════════════════════════════════════════════════════════
#  RESULTS TABLE + VERDICT
# ══════════════════════════════════════════════════════════════════════════
def print_table(results):
    width = 130
    print("\n" + "─" * width)
    print(f"{'Model':<28} {'Params':>9} {'Acc':>7} {'Prec(m)':>8} {'Rec(m)':>8} "
          f"{'F1(m)':>8} {'AUC(m)':>8} {'PeakMemMB':>10} {'Inf(ms)':>8}")
    print("─" * width)
    for r in results:
        mem = f"{r['peak_mem_mb']:.1f}" if r["peak_mem_mb"] is not None else "n/a"
        print(f"  {r['name']:<26} {r['params']:>9,} {r['test_acc']*100:>6.2f}% "
              f"{r['precision_macro']:>8.4f} {r['recall_macro']:>8.4f} "
              f"{r['f1_macro']:>8.4f} {r['roc_auc_macro']:>8.4f} {mem:>10} {r['inf_ms']:>7.2f}")
    print("─" * width)


def print_verdict(results):
    best = max(results, key=lambda r: r["test_acc"])
    print("\n" + "═" * 70)
    print("  FULL GRID ABLATION VERDICT")
    print("═" * 70)
    print(f"\n  Best accuracy: {best['name']} — {best['test_acc']*100:.2f}% "
          f"({best['params']:,} params)")
    print(f"  Acceptable accuracy-drop budget: {ACCEPTABLE_DROP_PP:.1f} pp\n")

    candidates = [r for r in results
                  if (best["test_acc"] - r["test_acc"]) * 100 <= ACCEPTABLE_DROP_PP]
    recommended = min(candidates, key=lambda r: r["params"])

    print(f"  RECOMMENDATION (smallest model within the accuracy-drop budget):")
    print(f"  -> {recommended['name']}")
    print(f"     {recommended['test_acc']*100:.2f}% accuracy "
          f"({(best['test_acc']-recommended['test_acc'])*100:.2f} pp below best)")
    print(f"     {recommended['params']:,} params "
          f"({(1 - recommended['params']/best['params'])*100:.0f}% fewer than the best model)")
    if recommended.get("peak_mem_mb") is not None and best.get("peak_mem_mb") is not None:
        print(f"     {recommended['peak_mem_mb']:.1f} MB peak activation memory "
              f"vs {best['peak_mem_mb']:.1f} MB for the best model")
    if recommended is best:
        print(f"\n  Note: the best-accuracy model is already the recommendation — every")
        print(f"  smaller model in the grid costs more than the {ACCEPTABLE_DROP_PP:.1f} pp budget allows.")
    print("\n" + "═" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("\nGalaxy Morphology — Full Grid Ablation (stem depth × S4D layers)")
    print(f"Device: {DEVICE}  |  Epochs: {EPOCHS}  |  LR: {LR}  |  Seed: {RNG_SEED}")
    print(f"Grid: {len(STEM_LADDER)} stem depths × {len(S4_LAYER_COUNTS)} S4D depths "
          f"= {len(STEM_LADDER)*len(S4_LAYER_COUNTS)} models, plus the raw S4D-only baseline\n")

    train_loader, val_loader, test_loader, X_test, y_test = load_and_split()
    in_ch = 3 if COLORED else 1

    results = []

    # ── [0] raw-pixel S4D-only baseline (no CNN stem at all) ─────────────
    set_seed()
    baseline = GalaxyClassifierS4D(s4_state=D_STATE, d_model=D_MODEL,
                                    num_classes=4, colored=COLORED)
    r = run_model("S4D-only baseline (no CNN)", baseline,
                  train_loader, val_loader, test_loader, X_test, y_test,
                  extra_info={"stem_layers": None, "num_s4_layers": 2})
    results.append(r)
    plot_confusion(r, os.path.join(OUT_DIR, "confusion_baseline_s4d_only.png"))
    plot_roc(r, os.path.join(OUT_DIR, "roc_baseline_s4d_only.png"))
    plot_training_curve(r, os.path.join(OUT_DIR, "training_baseline_s4d_only.png"))

    # ── [1..12] full grid: every stem depth × every S4D layer count ──────
    for stem_cls, n_s4 in itertools.product(STEM_LADDER, S4_LAYER_COUNTS):
        name = f"{stem_cls.NAME} + {n_s4} S4D"
        set_seed()
        model = GalaxyClassifierGrid(stem_cls, n_s4, in_ch)
        r = run_model(name, model, train_loader, val_loader, test_loader, X_test, y_test,
                      extra_info={"stem_layers": stem_cls.N_LAYERS, "num_s4_layers": n_s4})
        results.append(r)

        safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        plot_confusion(r, os.path.join(OUT_DIR, f"confusion_{safe}.png"))
        plot_roc(r, os.path.join(OUT_DIR, f"roc_{safe}.png"))
        plot_training_curve(r, os.path.join(OUT_DIR, f"training_{safe}.png"))

    # ── combined / summary plots ──────────────────────────────────────────
    print("\nGenerating summary plots …")
    plot_summary_bar(results, os.path.join(OUT_DIR, "accuracy_bar_all_models.png"))
    plot_accuracy_vs_params(results, os.path.join(OUT_DIR, "accuracy_vs_params.png"))
    plot_metrics_bar(results, os.path.join(OUT_DIR, "metrics_bar_all_models.png"))
    plot_noise_robustness(results, os.path.join(OUT_DIR, "noise_robustness_all_models.png"))
    plot_grid_heatmap(results, "test_acc", "Test Accuracy (%) — Stem Depth × S4D Layers",
                       os.path.join(OUT_DIR, "grid_heatmap_accuracy.png"), fmt=".1f", scale=100)
    plot_grid_heatmap(results, "params", "Parameter Count (K) — Stem Depth × S4D Layers",
                       os.path.join(OUT_DIR, "grid_heatmap_params.png"), fmt=".1f", scale=1/1000)
    if all(r.get("peak_mem_mb") is not None for r in results if r.get("stem_layers") is not None):
        plot_grid_heatmap(results, "peak_mem_mb", "Peak Activation Memory (MB) — Stem Depth × S4D Layers",
                           os.path.join(OUT_DIR, "grid_heatmap_peak_mem.png"), fmt=".1f")

    # ── results table + verdict ──────────────────────────────────────────
    print_table(results)
    print_verdict(results)

    # ── raw results dump (everything, for further analysis / your own plots) ──
    dump = [{k: v for k, v in r.items() if k not in ("roc_curves",)} for r in results]
    with open(os.path.join(OUT_DIR, "results_full_grid.json"), "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nFull results (incl. ROC curve points, per-epoch loss/acc) saved to:")
    print(f"  {os.path.join(OUT_DIR, 'results_full_grid.json')}")
    print(f"All plots saved to: {OUT_DIR}\n")

    # Weights + model summaries already went straight to KAGGLE_OUT_DIR as
    # each of the 13 models finished training; stage the JSON + every plot
    # from OUT_DIR there too so everything for this script lives in one
    # downloadable place instead of nested under ablation_results/full_grid/.
    kaggle_extras.stage_outputs(
        KAGGLE_OUT_DIR,
        os.path.join(OUT_DIR, "*.png"),
        os.path.join(OUT_DIR, "results_full_grid.json"),
    )


if __name__ == "__main__":
    main()