#!/usr/bin/env python
"""
Train GalaxyClassifierCNNS4D (CNN-stem -> S4D hybrid) at three parameter
scales and compare results.

Runs three variants of the hybrid model:
  ~100k params  (d_model=120, s4_state=60,  mid=32, 3 S4D layers) ->  98,332 params
  ~300k params  (d_model=112, s4_state=224, mid=80, 2 S4D layers) -> 298,868 params
  ~700k params  (d_model=192, s4_state=384, mid=80, 3 S4D layers) -> 699,748 params

All three use stem_reduction=16, colored=False (grayscale) so the only
axis that changes across the three runs is model capacity. The resulting
table, training curves, and confusion matrices let you see how accuracy
scales with param count for this architecture.

Run from the repo root:
    python scripts/train_hybrid_scale.py
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import GalaxyClassifierCNNS4D
from model.functions import load_data

import kaggle_extras
KAGGLE_OUT_DIR = kaggle_extras.output_dir("train_hybrid_scale")

# Reuse train() from scripts/train.py without re-running its guarded body
_spec = importlib.util.spec_from_file_location(
    "baseline_train_module", os.path.join(os.path.dirname(__file__), "train.py")
)
_baseline_train_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline_train_module)
train = _baseline_train_module.train

# -----------------------------------------------------------------------
# Config (shared across all three scale variants)
# -----------------------------------------------------------------------
RNG_SEED       = 42
BATCH_SIZE     = 64
LR             = 0.001
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTHING = 0.05
EPOCHS         = 40
COLORED        = False          # grayscale so scale is the only axis
CLASS_NAMES    = ["Smooth Round", "Smooth Cigar", "Edge-on Disk", "Unbarred Spiral"]
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------
# Three scale configurations
# Each tuple: (label, d_model, s4_state, mid_channels, num_s4_layers)
# -----------------------------------------------------------------------
SCALE_CONFIGS = [
    ("hybrid_100k",  120,  60,  32, 3),   #  98,332 params
    ("hybrid_300k",  112, 224,  80, 2),   # 298,868 params
    ("hybrid_700k",  192, 384,  80, 3),   # 699,748 params
]


class AugmentedGalaxyDataset(torch.utils.data.Dataset):
    """Random rotation + flip augmentation (train only)."""
    def __init__(self, X, y_onehot):
        self.X = X
        self.y_onehot = y_onehot

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img   = self.X[idx]
        label = self.y_onehot[idx]
        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k, dims=(1, 2))
        if random.random() < 0.5:
            img = torch.flip(img, dims=(2,))
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


def s4d_loop_ops(seq_len, d_model, d_state, num_layers):
    """Estimated S4D-loop ops for all stacked layers."""
    return num_layers * (seq_len * (d_state // 2) * d_model * 8)


def stem_conv_macs(model):
    """MACs contributed by the CNN stem's conv layers."""
    stem = getattr(model, "cnn_stem", None)
    if stem is None:
        return 0
    total = 0
    in_hw = 64
    for conv in (stem.stem_conv, stem.res_conv):
        k = conv.kernel_size[0]
        total += in_hw * in_hw * conv.out_channels * conv.in_channels * k * k
    for conv in (stem.down1, stem.down2):
        if conv is None:
            continue
        out_hw = in_hw // 2
        k = conv.kernel_size[0]
        total += out_hw * out_hw * conv.out_channels * conv.in_channels * k * k
        in_hw = out_hw
    return total


def total_est_ops(model, seq_len, d_state, num_layers):
    stem_macs = stem_conv_macs(model)
    proj_ops  = stem_macs if stem_macs else seq_len * model.hilbert_channels * model.d_model
    s4d_ops   = s4d_loop_ops(seq_len, model.d_model, d_state, num_layers)
    fc_ops    = model.d_model * model.fc.out_features
    return proj_ops + s4d_ops + fc_ops, s4d_ops


def measure_inference_time(model, colored, n_samples=100):
    model_cpu = model.to("cpu")
    model_cpu.eval()
    c = 3 if colored else 1
    x = torch.randn(1, c, 64, 64)
    with torch.no_grad():
        for _ in range(5):
            model_cpu(x)
        times = []
        for _ in range(n_samples):
            t0 = time.perf_counter()
            model_cpu(x)
            times.append(time.perf_counter() - t0)
    model.to(DEVICE)
    return float(np.mean(times))


def run_experiment(model, name, train_loader, val_loader, test_loader,
                   epochs, d_state, num_layers):
    set_seed(RNG_SEED)
    model = model.to(DEVICE)

    kaggle_extras.print_and_save_summary(
        model, (1, 3 if COLORED else 1, 64, 64), name, KAGGLE_OUT_DIR)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    n_params = count_params(model)
    print(f"\n{'='*70}")
    print(f"Training: {name}  ({n_params:,} params)")
    print(f"{'='*70}")

    t0   = time.time()
    hist = train(train_loader, val_loader, model, optimizer, loss_fn, epochs,
                 DEVICE, verbose=True, scheduler=scheduler)
    train_time = time.time() - t0

    # Test set evaluation
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs, return_logits=True)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(logits, dim=1)
            target = torch.argmax(labels, dim=1)
            correct += (preds == target).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    test_acc = correct / total
    cm       = confusion_matrix(all_targets, all_preds)

    # Same metrics the LaTeX report uses throughout its master results
    # table (this script originally only tracked accuracy + confusion matrix).
    metrics = kaggle_extras.compute_classification_metrics(
        all_targets, all_preds, all_probs, CLASS_NAMES)

    inf_time   = measure_inference_time(model, colored=COLORED, n_samples=100)
    seq_len    = getattr(model, "seq_len", 256)
    total_ops, s4d_ops = total_est_ops(model, seq_len, d_state, num_layers)

    weights_path = kaggle_extras.save_weights(model, name, KAGGLE_OUT_DIR)

    print(f"{name}: test_acc={test_acc*100:.2f}%  params={n_params:,}  "
          f"S4D-loop ops={s4d_ops:,}  train_time={train_time:.1f}s  "
          f"inference/sample={inf_time*1000:.3f}ms")
    print(f"  precision(macro)={metrics['precision_macro']:.4f}  "
          f"recall(macro)={metrics['recall_macro']:.4f}  "
          f"f1(macro)={metrics['f1_macro']:.4f}  "
          f"roc_auc(macro)={metrics['roc_auc_macro']:.4f}")

    return {
        "name":                         name,
        "n_params":                     n_params,
        "seq_len":                      seq_len,
        "test_acc":                     test_acc,
        "s4d_loop_ops":                 s4d_ops,
        "total_est_ops":                total_ops,
        "train_time_sec":               train_time,
        "inference_time_sec_per_sample": inf_time,
        "history":                      hist,
        "confusion_matrix":             cm.tolist(),
        "weights_path":                 weights_path,
        **metrics,
    }


def print_results_table(results):
    print("\n| Model | Params | Test Acc | F1 (macro) | Prec (macro) | Rec (macro) | "
          "AUC (macro) | Train time (s) | Inference/sample (ms) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['name']} | {r['n_params']:,} | {r['test_acc']*100:.2f}% | "
              f"{r['f1_macro']:.4f} | {r['precision_macro']:.4f} | {r['recall_macro']:.4f} | "
              f"{r['roc_auc_macro']:.4f} | {r['train_time_sec']:.1f} | "
              f"{r['inference_time_sec_per_sample']*1000:.3f} |")


def plot_training_curves(results, out_path="training_curves_scale.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        label = f"{r['name']} ({r['n_params']:,})"
        axes[0].plot(r["history"]["loss"],         label=label)
        axes[1].plot(r["history"]["val_accuracy"],  label=label)
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


def plot_accuracy_vs_params(results, out_path="accuracy_vs_params_scale.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    params = [r["n_params"] for r in results]
    accs   = [r["test_acc"] * 100 for r in results]
    names  = [r["name"] for r in results]
    ax.plot(params, accs, marker="o", linewidth=2)
    for x, y, n in zip(params, accs, names):
        ax.annotate(f"{n}\n{y:.2f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs parameter count (CNN-stem + S4D hybrid)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def plot_confusion_matrices(results, out_path="confusion_matrices_scale.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(f"{r['name']}  ({r['n_params']:,} params)\nTest Acc: {r['test_acc']*100:.2f}%")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def main():
    set_seed(RNG_SEED)
    print(f"Using device: {DEVICE}")
    print(f"\nParameter-scale sweep: {[c[0] for c in SCALE_CONFIGS]}")

    # ---- Data ----
    X, y_onehot, y = load_data(root="./data", download=True, train=True, colored=COLORED)
    NUM_CLASSES    = y_onehot.shape[1]

    x_train, x_val, y_train_oh, y_val_oh = train_test_split(
        X, y_onehot, test_size=0.2, random_state=RNG_SEED, stratify=y
    )
    train_ds     = AugmentedGalaxyDataset(x_train, y_train_oh)
    val_ds       = TensorDataset(x_val,   y_val_oh)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    X_test, y_test_oh, _ = load_data(root="./data", download=True, train=False, colored=COLORED)
    test_ds      = TensorDataset(X_test, y_test_oh)
    test_loader  = DataLoader(test_ds,  batch_size=64)

    # ---- Print actual param counts before training ----
    print("\nModel summary:")
    for label, d_model, s4_state, mid, num_s4 in SCALE_CONFIGS:
        m = GalaxyClassifierCNNS4D(
            d_model=d_model, s4_state=s4_state, mid_channels=mid,
            num_s4_layers=num_s4, stem_reduction=16, colored=COLORED,
            num_classes=NUM_CLASSES,
        )
        print(f"  {label}: d_model={d_model}, s4_state={s4_state}, "
              f"mid={mid}, {num_s4}L  ->  {count_params(m):,} params")

    # ---- Train all three ----
    results = []
    for label, d_model, s4_state, mid, num_s4 in SCALE_CONFIGS:
        model = GalaxyClassifierCNNS4D(
            d_model=d_model, s4_state=s4_state, mid_channels=mid,
            num_s4_layers=num_s4, stem_reduction=16, colored=COLORED,
            num_classes=NUM_CLASSES,
        )
        results.append(run_experiment(
            model, label, train_loader, val_loader, test_loader,
            EPOCHS, d_state=s4_state, num_layers=num_s4,
        ))

    # ---- Report ----
    print_results_table(results)

    out_json = "results_scale_sweep.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")

    plot_training_curves(results,     out_path="training_curves_scale.png")
    plot_accuracy_vs_params(results,  out_path="accuracy_vs_params_scale.png")
    plot_confusion_matrices(results,  out_path="confusion_matrices_scale.png")

    # Weights + model summaries already went straight to KAGGLE_OUT_DIR as
    # each scale variant finished training; stage the JSON + plots there
    # too so everything for this script lives in one downloadable place.
    kaggle_extras.stage_outputs(
        KAGGLE_OUT_DIR,
        "results_scale_sweep.json",
        "training_curves_scale.png",
        "accuracy_vs_params_scale.png",
        "confusion_matrices_scale.png",
    )


if __name__ == "__main__":
    main()