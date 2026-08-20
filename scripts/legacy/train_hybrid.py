#!/usr/bin/env python
"""
Train GalaxyClassifierCNNS4D (CNN-stem -> S4D hybrid) only, and dump the
metrics/plots called for in the experiment writeup.

This version intentionally does NOT run the S4D baseline at all -- neither
retraining it nor loading its checkpoint -- for a fast, single-model pass.
If you want the baseline back in for the full comparison, see the
"add baseline back in" note near the bottom of main().

Run from the repo root:
    python scripts/train_hybrid.py

Config below is copied from scripts/train.py's baseline run (same RNG_SEED,
BATCH_SIZE, loss, split, COLORED) so a future comparison against the
baseline stays close to apples-to-apples. Note LR and the addition of a
CosineAnnealingLR scheduler are deviations from the original baseline
training regime -- see the LR/scheduler comments below for why, and account
for that if comparing directly against a baseline trained without them.

This script does NOT duplicate the training loop: it imports train() from
scripts/train.py directly (by file path, so it works regardless of how this
script is invoked, and without triggering that file's own notebook-style
body, which is guarded behind `if __name__ == "__main__":`).
"""
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Make sure the repo root is importable regardless of cwd (mirrors the
# sys.path handling already done in model/__init__.py)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import GalaxyClassifierCNNS4D
from model.functions import load_data

import kaggle_extras
KAGGLE_OUT_DIR = kaggle_extras.output_dir("train_hybrid")

# --- Reuse train() from scripts/train.py unmodified, without re-running ---
# --- that file's own guarded (if __name__ == "__main__") script body.   ---
_spec = importlib.util.spec_from_file_location(
    "baseline_train_module", os.path.join(os.path.dirname(__file__), "train.py")
)
_baseline_train_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline_train_module)
train = _baseline_train_module.train

# ---------------------------------------------------------------------
# Config -- RESEARCH PHASE (post-course, optimizing purely for accuracy)
# ---------------------------------------------------------------------
RNG_SEED = 42
# Bumped from 16: the CNN stem cuts seq_len 16x/4x, so S4D's O(L) memory
# blowup that forced BATCH_SIZE=16 in the course baseline no longer
# applies here. Larger batches stabilize gradients on a small (~8k image)
# dataset.
BATCH_SIZE = 64
LR = 0.001            # slightly lower than the course's 0.0015; the model
                       # has more capacity now (bigger stem, GroupNorm,
                       # optional 3rd S4D layer) and a lower LR + weight
                       # decay is a safer combination for that.
WEIGHT_DECAY = 1e-4    # L2 regularization -- wasn't used at all in the
                       # course version. Helps now that capacity is up.
LABEL_SMOOTHING = 0.05 # softens targets slightly; tends to help most on
                       # genuinely ambiguous adjacent classes (exactly
                       # the Smooth Cigar / Edge-on Disk pair we're stuck on)
EPOCHS = 40            # doubled: bigger model + regularization typically
                       # needs a longer schedule to fully converge
COLORED = False        # <<< TEMPORARILY back to grayscale, on purpose.
                       # This isolates the color contribution: same
                       # redesigned stem (stride-1 full-res extraction,
                       # GroupNorm, more capacity) + same training recipe
                       # (AdamW, weight decay, label smoothing, 40 epochs)
                       # as the 86.80% color run -- only COLORED changes.
                       # If this lands back near the old ~69% ceiling, it
                       # confirms color (not the stem/training changes)
                       # was doing essentially all the work. If it lands
                       # meaningfully above 69% but below 86.80%, the stem
                       # redesign was worth some of the gain on its own.
CLASS_NAMES = ["Smooth Round", "Smooth Cigar", "Edge-on Disk", "Unbarred Spiral"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_MODEL = 64
D_STATE = 64
NUM_S4_LAYERS = 2      # winning config from the color runs (86.80%
                       # beat 86.65% at 3 layers) -- kept at 2 here so
                       # this run isolates color, not depth. (see
                       # STEM_CONFIGS below for where this is used)

class AugmentedGalaxyDataset(torch.utils.data.Dataset):
    """
    Wraps a (X, y) tensor pair and applies random label-preserving
    augmentation on each __getitem__ call: random 90/180/270 rotation and
    random horizontal/vertical flip. Galaxies have no canonical orientation,
    so these transforms don't change the true class -- they're close to
    free additional training signal for a small (8k image) dataset.

    Train-only: never wrap val/test sets with this, since evaluation needs
    to reflect real, unaugmented performance.
    """
    def __init__(self, X, y_onehot):
        self.X = X
        self.y_onehot = y_onehot

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  # (C, H, W)
        label = self.y_onehot[idx]

        # Random rotation: 0, 90, 180, or 270 degrees
        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k, dims=(1, 2))

        # Random horizontal flip
        if random.random() < 0.5:
            img = torch.flip(img, dims=(2,))

        # Random vertical flip
        if random.random() < 0.5:
            img = torch.flip(img, dims=(1,))

        return img, label

def set_seed(seed=RNG_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def s4d_loop_ops(seq_len, d_model=D_MODEL, d_state=D_STATE):
    """
    Combined op count of both stacked S4D layers (s4_1, s4_2), per the
    FLOPS comment in model/gclassifier.py:
        2 * (seq_len * (d_state // 2) * d_model * 8)
    The leading 2 accounts for the two stacked layers; the trailing 8 is
    ~2 complex MACs per state element per step, at ~4 real ops/complex MAC.
    """
    return 2 * (seq_len * (d_state // 2) * d_model * 8)


def stem_conv_macs(model):
    """
    Sum over the CNN stem's conv layers of out_H*out_W*out_C*in_C*k*k.
    Updated for the research-version CNNStem, which has 4 possible conv
    layers (stem_conv, res_conv -- both stride 1, full-res; down1, down2
    -- stride 2, only down2 present when reduction==16).
    """
    stem = getattr(model, "cnn_stem", None)
    if stem is None:
        return 0
    total = 0
    in_hw = 64
    # stem_conv, res_conv: stride 1, resolution unchanged (64x64)
    for conv in (stem.stem_conv, stem.res_conv):
        k = conv.kernel_size[0]
        total += in_hw * in_hw * conv.out_channels * conv.in_channels * k * k
    # down1 (and down2 if present): stride 2, halves resolution each time
    for conv in (stem.down1, stem.down2):
        if conv is None:
            continue
        out_hw = in_hw // 2
        k = conv.kernel_size[0]
        total += out_hw * out_hw * conv.out_channels * conv.in_channels * k * k
        in_hw = out_hw
    return total


def total_est_ops(model, seq_len):
    """
    Total estimated ops per forward pass: stem conv MACs (no separate
    uproject -- the stem's last conv already projects to d_model) +
    S4D-loop ops + classifier head.
    """
    stem_macs = stem_conv_macs(model)
    if stem_macs == 0:
        proj_ops = seq_len * model.hilbert_channels * model.d_model
    else:
        proj_ops = stem_macs
    s4d_ops = s4d_loop_ops(seq_len, d_model=model.d_model)
    fc_ops = model.d_model * model.fc.out_features
    return proj_ops + s4d_ops + fc_ops, s4d_ops


def measure_inference_time(model, colored, n_samples=100):
    """Mean per-sample CPU inference time (eval mode, batch size 1)."""
    model_cpu = model.to("cpu")
    model_cpu.eval()
    c = 3 if colored else 1
    x = torch.randn(1, c, 64, 64)
    with torch.no_grad():
        for _ in range(5):  # warmup
            model_cpu(x)
        times = []
        for _ in range(n_samples):
            t0 = time.perf_counter()
            model_cpu(x)
            times.append(time.perf_counter() - t0)
    model.to(DEVICE)
    return float(np.mean(times))


def run_experiment(model, name, train_loader, val_loader, test_loader, epochs):
    set_seed(RNG_SEED)  # re-seed before each run so init/shuffling is reproducible
    model = model.to(DEVICE)

    kaggle_extras.print_and_save_summary(
        model, (1, 3 if COLORED else 1, 64, 64), name, KAGGLE_OUT_DIR)

    # AdamW (decoupled weight decay) instead of plain Adam -- with
    # WEIGHT_DECAY=0 this is identical to Adam, but decoupled decay is the
    # correct choice now that WEIGHT_DECAY > 0 (plain Adam's L2-via-grad
    # interacts badly with its per-parameter adaptive LR).
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Cosine-anneal LR from LR down toward 0 over the run, so late-epoch
    # steps shrink instead of staying fixed-size -- fixes the oscillation
    # seen near convergence with a constant LR (see conversation notes).
    # T_max=epochs completes exactly one decay cycle over the full run.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"\n{'='*70}\nTraining: {name}\n{'='*70}")
    t0 = time.time()
    hist = train(train_loader, val_loader, model, optimizer, loss_fn, epochs, DEVICE,
                 verbose=True, scheduler=scheduler)
    train_time = time.time() - t0

    # Held-out test set evaluation
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs, return_logits=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            target = torch.argmax(labels, dim=1)
            correct += (preds == target).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    test_acc = correct / total
    cm = confusion_matrix(all_targets, all_preds)

    # Same metrics the LaTeX report uses throughout its master results
    # table (this script originally only tracked accuracy + confusion matrix).
    metrics = kaggle_extras.compute_classification_metrics(
        all_targets, all_preds, all_probs, CLASS_NAMES)

    inf_time = measure_inference_time(model, colored=COLORED, n_samples=100)
    n_params = count_params(model)
    seq_len = getattr(model, "seq_len", 256)
    total_ops, s4d_ops = total_est_ops(model, seq_len)

    weights_path = kaggle_extras.save_weights(model, name, KAGGLE_OUT_DIR)

    print(f"{name}: test_acc={test_acc*100:.2f}%  params={n_params:,}  "
          f"S4D-loop ops={s4d_ops:,}  train_time={train_time:.1f}s  "
          f"inference/sample={inf_time*1000:.3f}ms")
    print(f"  precision(macro)={metrics['precision_macro']:.4f}  "
          f"recall(macro)={metrics['recall_macro']:.4f}  "
          f"f1(macro)={metrics['f1_macro']:.4f}  "
          f"roc_auc(macro)={metrics['roc_auc_macro']:.4f}")

    return {
        "name": name,
        "seq_len": seq_len,
        "params": n_params,
        "test_acc": test_acc,
        "s4d_loop_ops": s4d_ops,
        "total_est_ops": total_ops,
        "train_time_sec": train_time,
        "inference_time_sec_per_sample": inf_time,
        "history": hist,
        "confusion_matrix": cm.tolist(),
        "weights_path": weights_path,
        **metrics,
    }


def print_results_table(results):
    print("\n| Model | Seq Len | Params | Test Acc | F1 (macro) | Prec (macro) | Rec (macro) | "
          "AUC (macro) | Train time (s) | Inference/sample (ms) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['name']} | {r['seq_len']} | {r['params']:,} | {r['test_acc']*100:.2f}% | "
              f"{r['f1_macro']:.4f} | {r['precision_macro']:.4f} | {r['recall_macro']:.4f} | "
              f"{r['roc_auc_macro']:.4f} | {r['train_time_sec']:.1f} | "
              f"{r['inference_time_sec_per_sample']*1000:.3f} |")


def plot_training_curves(results, out_path="training_curves_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        axes[0].plot(r["history"]["loss"], label=r["name"])
        axes[1].plot(r["history"]["val_accuracy"], label=r["name"])
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def plot_confusion_matrices(results, out_path="confusion_matrices_comparison.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(f"{r['name']}\nTest Acc: {r['test_acc']*100:.2f}%")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def main():
    set_seed(RNG_SEED)
    print(f"Using device: {DEVICE}")

    X, y_onehot, y = load_data(root="./data", download=True, train=True, colored=COLORED)
    NUM_CLASSES = y_onehot.shape[1]

    x_train, x_val, y_train_onehot, y_val_onehot = train_test_split(
        X, y_onehot, test_size=0.2, random_state=RNG_SEED, stratify=y
    )
    train_ds = AugmentedGalaxyDataset(x_train, y_train_onehot)  # was: TensorDataset(x_train, y_train_onehot)
    val_ds = TensorDataset(x_val, y_val_onehot)  # unchanged -- no augmentation at eval time
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    X_test, y_test_onehot, y_test = load_data(root="./data", download=True, train=False, colored=COLORED)
    test_ds = TensorDataset(X_test, y_test_onehot)
    test_loader = DataLoader(test_ds, batch_size=64)

    results = []

    # --- Ablation: CNN stem (16x) + S4D (seq_len=256), GRAYSCALE ---
    # Same redesigned stem + same training recipe as the 86.80% color run
    # -- only COLORED changed (True -> False here). Isolates how much of
    # the gain came from color vs. the architecture/training changes.
    hybrid16_gray = GalaxyClassifierCNNS4D(
        num_classes=NUM_CLASSES, colored=COLORED, stem_reduction=16,
        num_s4_layers=NUM_S4_LAYERS,
    )
    results.append(run_experiment(
        hybrid16_gray, f"CNN stem (16x, grayscale) + S4D (seq_len=256, {NUM_S4_LAYERS} layers)",
        train_loader, val_loader, test_loader, EPOCHS,
    ))

    # --- For reference (not re-run here) ---
    # CNN stem 16x, grayscale, course-era stem/recipe:       69.05% / 69.10%
    # CNN stem 16x, color,    redesigned stem/recipe, 2 layer: 86.80%
    # CNN stem 16x, color,    redesigned stem/recipe, 3 layer: 86.65%
    #
    # --- Add the baseline back in later for the full comparison ---
    # from model import GalaxyClassifierS4D
    # baseline = GalaxyClassifierS4D(num_classes=NUM_CLASSES, colored=COLORED)
    # baseline.load_state_dict(torch.load(
    #     os.path.join(_REPO_ROOT, "model_params", "galaxys4-30EPOCH-STANDARD.pth"),
    #     map_location=DEVICE,
    # ))
    # ... (evaluate on test_loader, same pattern as run_experiment's eval block,
    #      then results.insert(0, {...}) so it prints first in the table)
    # NOTE: that checkpoint was trained on grayscale input (COLORED=False),
    # so it IS directly comparable to this run (both grayscale) if you want
    # an apples-to-apples baseline-vs-hybrid comparison at this point.

    print_results_table(results)

    with open("results_table_16x_grayscale_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_table_16x_grayscale_ablation.json")

    plot_training_curves(results, out_path="training_curves_16x_grayscale_ablation.png")
    plot_confusion_matrices(results, out_path="confusion_matrices_16x_grayscale_ablation.png")

    # Weights + model summaries already went straight to KAGGLE_OUT_DIR as
    # each model finished training; stage the JSON + plots there too so
    # everything for this script lives in one downloadable place.
    kaggle_extras.stage_outputs(
        KAGGLE_OUT_DIR,
        "results_table_16x_grayscale_ablation.json",
        "training_curves_16x_grayscale_ablation.png",
        "confusion_matrices_16x_grayscale_ablation.png",
    )


if __name__ == "__main__":
    main()
