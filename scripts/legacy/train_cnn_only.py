#!/usr/bin/env python
"""
CNN-only ablation: answers the TA's four questions directly.

  Q1. Would a ~43K-param CNN alone give similar accuracy to the
      ~55K-param CNN+S4D hybrid (86.80%)?
  Q2. Would a ~60K-param CNN alone give similar accuracy?
  Q3. Is S4D actually enhancing the CNN's signal, or is the CNN
      doing all the work and S4D just along for the ride?
  Q4. How small can the CNN get before accuracy collapses?

Strategy
--------
Run four models, all at the same training config (color, 40 epochs,
AdamW, weight decay, label smoothing -- identical to the 86.80% run):

  - Tiny CNN   (~10K params): deliberately undersized lower-bound
                               for Q4 -- where does it fall apart?
  - Small CNN  (~43K params): matched to the hybrid's 55K but with
                               NO S4D layers at all
  - Large CNN  (~61K params): matched to the 3-layer hybrid's 63K
  - Hybrid ref (55K params):  the existing winning config, re-run
                               here so all four numbers come from the
                               same session (same data splits, same
                               RNG seed) for a fair comparison.

If Small CNN (~43K) and Large CNN (~61K) land near 86.80%:
  -> S4D wasn't adding much; the CNN stem was doing the work.
If they land well below 86.80%:
  -> S4D genuinely enhances the signal the CNN provides.
The Tiny CNN result shows where the capacity floor actually is.

Run from the repo root:
    python scripts/train_cnn_only.py
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model import GalaxyClassifierCNNS4D, GalaxyClassifierCNNOnly
from model.functions import load_data

import kaggle_extras
KAGGLE_OUT_DIR = kaggle_extras.output_dir("train_cnn_only")

# Reuse train() from scripts/train.py unmodified
_spec = importlib.util.spec_from_file_location(
    "baseline_train_module", os.path.join(os.path.dirname(__file__), "train.py")
)
_baseline_train_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline_train_module)
train = _baseline_train_module.train

# ---------------------------------------------------------------------
# Config -- identical to the 86.80% run so the comparison is fair
# ---------------------------------------------------------------------
RNG_SEED    = 42
BATCH_SIZE  = 64
LR          = 0.001
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.05
EPOCHS      = 40
COLORED     = True
CLASS_NAMES = ["Smooth Round", "Smooth Cigar", "Edge-on Disk", "Unbarred Spiral"]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# Model configs: (label, mid_channels, d_model, use_refine_conv, note)
# ---------------------------------------------------------------------
# Parameter counts (precomputed -- see param-count script at top of file):
#   Tiny:   mid=16, d_model=32, refine=False  ->  ~10K params
#   Small:  mid=32, d_model=64, refine=True   ->  ~43K params
#   Large:  mid=40, d_model=70, refine=True   ->  ~61K params
# Hybrid reference: the winning CNN+S4D (55K, 86.80%)
CNN_CONFIGS = [
    # (name,                  mid, d_model, refine, note)
    ("CNN-only Tiny  (~10K)", 16,  32,      False,  "lower-bound: where does accuracy collapse?"),
    ("CNN-only Small (~43K)", 32,  64,      True,   "matched to hybrid's ~55K param budget"),
    ("CNN-only Large (~61K)", 40,  70,      True,   "matched to 3-layer hybrid's ~63K param budget"),
]

# ---------------------------------------------------------------------
# Helpers (copied from train_hybrid.py so this script is self-contained)
# ---------------------------------------------------------------------
class AugmentedGalaxyDataset(torch.utils.data.Dataset):
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

def measure_inference_time(model, n_samples=100):
    model_cpu = model.to("cpu").eval()
    c = 3 if COLORED else 1
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

def run_experiment(model, name, train_loader, val_loader, test_loader, epochs):
    set_seed(RNG_SEED)
    model = model.to(DEVICE)

    kaggle_extras.print_and_save_summary(
        model, (1, 3 if COLORED else 1, 64, 64), name, KAGGLE_OUT_DIR)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(f"\n{'='*70}\nTraining: {name}\n{'='*70}")
    t0   = time.time()
    hist = train(train_loader, val_loader, model, optimizer, loss_fn,
                 epochs, DEVICE, verbose=True, scheduler=scheduler)
    train_time = time.time() - t0

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

    inf_time = measure_inference_time(model)
    n_params = count_params(model)

    # ---- precision / recall / f1 (per-class + macro + weighted) ----
    n_classes = len(CLASS_NAMES)
    labels_arr = list(range(n_classes))
    prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels_arr, average=None, zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels_arr, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels_arr, average="weighted", zero_division=0
    )

    # ---- ROC-AUC (one-vs-rest, needs class probabilities) ----
    y_true_bin  = label_binarize(all_targets, classes=labels_arr)
    probs_arr   = np.array(all_probs)
    roc_per_class = {}
    for i, cname_cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs_arr[:, i])
        roc_per_class[cname_cls] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))
        }
    try:
        auc_macro = float(roc_auc_score(y_true_bin, probs_arr, average="macro", multi_class="ovr"))
        auc_weighted = float(roc_auc_score(y_true_bin, probs_arr, average="weighted", multi_class="ovr"))
    except ValueError:
        # Can happen if a class is missing from the test batch
        auc_macro, auc_weighted = float("nan"), float("nan")

    weights_path = kaggle_extras.save_weights(model, name, KAGGLE_OUT_DIR)

    print(f"{name}: test_acc={test_acc*100:.2f}%  params={n_params:,}  "
          f"train_time={train_time:.1f}s  inference/sample={inf_time*1000:.3f}ms")
    print(f"  precision(macro)={prec_macro:.4f}  recall(macro)={rec_macro:.4f}  "
          f"f1(macro)={f1_macro:.4f}  roc_auc(macro)={auc_macro:.4f}")
    for i, cname_cls in enumerate(CLASS_NAMES):
        print(f"    [{cname_cls:<16}] precision={prec_c[i]:.3f}  recall={rec_c[i]:.3f}  "
              f"f1={f1_c[i]:.3f}  support={int(support_c[i])}  "
              f"auc={roc_per_class[cname_cls]['auc']:.3f}")

    return {
        "name":        name,
        "params":      n_params,
        "test_acc":    test_acc,
        "train_time_sec": train_time,
        "inference_time_sec_per_sample": inf_time,
        "history":     hist,
        "confusion_matrix": cm.tolist(),
        "weights_path": weights_path,
        "precision_macro": float(prec_macro),
        "recall_macro":    float(rec_macro),
        "f1_macro":        float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted":    float(rec_weighted),
        "f1_weighted":        float(f1_weighted),
        "per_class": {
            cname_cls: {
                "precision": float(prec_c[i]),
                "recall":    float(rec_c[i]),
                "f1":        float(f1_c[i]),
                "support":   int(support_c[i]),
                "roc_auc":   roc_per_class[cname_cls]["auc"],
            }
            for i, cname_cls in enumerate(CLASS_NAMES)
        },
        "roc_auc_macro":    auc_macro,
        "roc_auc_weighted": auc_weighted,
        "roc_curves": roc_per_class,
    }

def print_results_table(results):
    print("\n| Model | Params | Test Acc | Precision (macro) | Recall (macro) | "
          "F1 (macro) | ROC-AUC (macro) | Train time (s) | Inference/sample (ms) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['name']} | {r['params']:,} | {r['test_acc']*100:.2f}% | "
              f"{r['precision_macro']:.4f} | {r['recall_macro']:.4f} | "
              f"{r['f1_macro']:.4f} | {r['roc_auc_macro']:.4f} | "
              f"{r['train_time_sec']:.1f} | {r['inference_time_sec_per_sample']*1000:.3f} |")

def plot_training_curves(results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        axes[0].plot(r["history"]["loss"], label=r["name"])
        axes[1].plot(r["history"]["val_accuracy"], label=r["name"])
    axes[0].set_title("Training loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(fontsize=7)
    axes[1].set_title("Validation accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); print(f"Saved {out_path}")

def plot_roc_curves(results, out_path):
    """One subplot per model, one-vs-rest ROC curve per class + macro AUC."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        for cname_cls, curve in r["roc_curves"].items():
            ax.plot(curve["fpr"], curve["tpr"],
                    label=f"{cname_cls} (AUC={curve['auc']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_title(f"{r['name']}\nMacro AUC: {r['roc_auc_macro']:.3f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")

def plot_confusion_matrices(results, out_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1: axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(f"{r['name']}\nTest Acc: {r['test_acc']*100:.2f}%")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); print(f"Saved {out_path}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    set_seed(RNG_SEED)
    print(f"Using device: {DEVICE}")

    X, y_onehot, y = load_data(root="./data", download=True, train=True, colored=COLORED)
    NUM_CLASSES = y_onehot.shape[1]

    x_train, x_val, y_tr, y_val = train_test_split(
        X, y_onehot, test_size=0.2, random_state=RNG_SEED, stratify=y
    )
    train_ds     = AugmentedGalaxyDataset(x_train, y_tr)
    val_ds       = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    X_test, y_test_onehot, _ = load_data(root="./data", download=True, train=False, colored=COLORED)
    test_loader = DataLoader(TensorDataset(X_test, y_test_onehot), batch_size=64)

    results = []

    # ---- CNN-only variants (Q1, Q2, Q4) ----
    for cname, mid, dmodel, refine, note in CNN_CONFIGS:
        print(f"\n>>> {cname}  [{note}]")
        model = GalaxyClassifierCNNOnly(
            num_classes=NUM_CLASSES,
            colored=COLORED,
            stem_reduction=16,
            mid_channels=mid,
            d_model=dmodel,
            use_refine_conv=refine,
        )
        print(f"    Params: {count_params(model):,}")
        results.append(run_experiment(
            model, cname, train_loader, val_loader, test_loader, EPOCHS
        ))

    # ---- Hybrid reference (Q3: is S4D adding anything?) ----
    # Re-run the winning config in the same session so the comparison is
    # same splits, same seed -- not "our old run vs a new run."
    print("\n>>> Hybrid reference: CNN stem (16x, color) + S4D (seq_len=256, 2 layers)")
    hybrid = GalaxyClassifierCNNS4D(
        num_classes=NUM_CLASSES, colored=COLORED,
        stem_reduction=16, num_s4_layers=2,
    )
    print(f"    Params: {count_params(hybrid):,}")
    results.append(run_experiment(
        hybrid, "CNN+S4D hybrid (16x, color, 2 layers) -- reference",
        train_loader, val_loader, test_loader, EPOCHS,
    ))

    print_results_table(results)

    with open("results_cnn_only_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results_cnn_only_ablation.json")

    plot_training_curves(results,     out_path="training_curves_cnn_only.png")
    plot_confusion_matrices(results,  out_path="confusion_matrices_cnn_only.png")
    plot_roc_curves(results,          out_path="roc_curves_cnn_only.png")

    # Weights + model summaries already went straight to KAGGLE_OUT_DIR as
    # each model finished training; stage the JSON + plots there too so
    # everything for this script lives in one downloadable place.
    kaggle_extras.stage_outputs(
        KAGGLE_OUT_DIR,
        "results_cnn_only_ablation.json",
        "training_curves_cnn_only.png",
        "confusion_matrices_cnn_only.png",
        "roc_curves_cnn_only.png",
    )

    # ---- Print a direct answer to each TA question ----
    print("\n" + "="*70)
    print("ANSWERS TO TA QUESTIONS")
    print("="*70)
    names    = [r["name"]     for r in results]
    accs     = [r["test_acc"] for r in results]
    params   = [r["params"]   for r in results]
    hybrid_acc = accs[-1]

    print(f"\nQ1. ~43K CNN alone vs hybrid (86.80%): {accs[1]*100:.2f}%  "
          f"(gap: {(hybrid_acc - accs[1])*100:+.2f} pp)")
    print(f"Q2. ~61K CNN alone vs hybrid (86.80%): {accs[2]*100:.2f}%  "
          f"(gap: {(hybrid_acc - accs[2])*100:+.2f} pp)")
    print(f"Q3. Hybrid re-run:                     {hybrid_acc*100:.2f}%")
    verdict = "S4D IS adding signal (CNN alone is weaker)" \
              if (hybrid_acc - accs[1]) > 0.01 \
              else "S4D is NOT clearly adding signal (CNN alone matches)"
    print(f"    -> {verdict}")
    print(f"Q4. Tiny CNN (~10K) lower bound:       {accs[0]*100:.2f}%")

if __name__ == "__main__":
    main()
