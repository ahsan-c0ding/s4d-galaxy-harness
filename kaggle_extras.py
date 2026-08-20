"""
kaggle_extras.py -- not part of the original s4d.zip.

Shared helpers added for the Kaggle notebook run:
  - a torchinfo model summary for every model trained (architecture +
    per-layer param counts), printed and saved alongside each run's
    other artifacts;
  - the same classification metrics the LaTeX report uses throughout its
    master results table (accuracy, precision/recall/F1 macro, one-vs-rest
    ROC-AUC macro), for the two scripts (train_hybrid.py,
    train_hybrid_scale.py) that didn't already compute them;
  - staging of weights, training curves, and confusion matrices into a
    flat, easy-to-find/download location directly under /kaggle/working/,
    instead of nested inside this repo's own working directory.
"""
import glob
import os
import re
import shutil

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize

KAGGLE_ROOT = "/kaggle/working"
OUTPUTS_ROOT = os.path.join(KAGGLE_ROOT, "outputs")


def safe_slug(name):
    """Turn a human-readable model name into a filesystem-safe slug."""
    s = re.sub(r"[^\w\-.]+", "_", name.strip())
    return re.sub(r"_+", "_", s).strip("_")


def output_dir(script_tag):
    """/kaggle/working/outputs/<script_tag>/ -- created on first use.

    Falls back to a local ./outputs/<script_tag>/ if /kaggle/working isn't
    writable (e.g. testing outside Kaggle), so a script never crashes
    purely because it isn't running in a Kaggle kernel.
    """
    d = os.path.join(OUTPUTS_ROOT, script_tag)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = os.path.join("outputs", script_tag)
        os.makedirs(d, exist_ok=True)
    return d


def print_and_save_summary(model, input_size, name, out_dir):
    """Run torchinfo.summary(), print it, and save the text alongside the
    other artifacts for this run. Falls back to a plain parameter count if
    torchinfo isn't importable for some reason, rather than crashing a
    training run over a reporting nicety."""
    header = f"\n{'=' * 70}\nModel summary: {name}\n{'=' * 70}"
    print(header)
    try:
        from torchinfo import summary as _summary
        stats = _summary(model, input_size=input_size, verbose=0,
                          col_names=("input_size", "output_size", "num_params"))
        text = str(stats)
    except Exception as exc:  # pragma: no cover
        n_params = sum(p.numel() for p in model.parameters())
        text = f"(torchinfo summary unavailable: {exc})\nTotal params: {n_params:,}"
    print(text)
    with open(os.path.join(out_dir, f"{safe_slug(name)}_summary.txt"), "w") as f:
        f.write(header + "\n" + text + "\n")
    return text


def save_weights(model, name, out_dir):
    """Save model.state_dict() so trained weights survive past the Kaggle
    session, not just the in-memory results dict."""
    path = os.path.join(out_dir, f"{safe_slug(name)}_weights.pth")
    torch.save(model.state_dict(), path)
    print(f"Saved weights -> {path}")
    return path


def compute_classification_metrics(all_targets, all_preds, all_probs, class_names):
    """Precision/recall/F1 (macro) + one-vs-rest ROC-AUC (macro) -- the
    same metrics used throughout the LaTeX report's master results table
    (Acc / F1 / Prec / Rec / AUC columns)."""
    labels_arr = list(range(len(class_names)))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels_arr, average="macro", zero_division=0
    )
    y_true_bin = label_binarize(all_targets, classes=labels_arr)
    probs_arr = np.array(all_probs)
    try:
        auc_macro = float(roc_auc_score(y_true_bin, probs_arr, average="macro", multi_class="ovr"))
    except ValueError:
        # can happen if a class is entirely absent from the test batch
        auc_macro = float("nan")
    return {
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "roc_auc_macro": auc_macro,
    }


def stage_outputs(out_dir, *patterns):
    """Copy every file matching each glob pattern into out_dir. Copies,
    not moves -- originals stay where the script originally wrote them
    too, in case anything else in this notebook still reads them from
    their original relative path (the results-display cells do)."""
    copied = []
    for pattern in patterns:
        for src in glob.glob(pattern):
            dst = os.path.join(out_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            copied.append(dst)
    print(f"Staged {len(copied)} file(s) -> {out_dir}")
    return copied
