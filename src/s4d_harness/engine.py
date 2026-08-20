"""
The shared train/evaluate loop every architecture in this repo now goes
through, regardless of family or recipe.

This is the single most important file in the harness: before it existed,
each of the 15 original notebooks had its own copy-pasted training loop,
which is how training recipe became an undetected confound for months (see
the LaTeX report's "Training Recipe as a Confound" section, and the recipe
mismatch this project's own controlled study found: 8-10pp one direction,
2.25-2.55pp the other). Using this module for every future run makes that
class of bug structurally harder to reintroduce, because there is now only
one training loop to get right instead of fifteen.

Adapted directly from cell 29 of notebooks/report-discrepency-testing.ipynb
(``evaluate_model`` / ``macro_metrics`` / ``run_one``), generalized so
``build_model`` is passed in rather than hardcoded to the six controlled-
study architectures, and so dataset paths aren't Kaggle-specific.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .recipes import RECIPES, make_loaders, make_optimizer_and_scheduler, set_all_seeds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Smooth Round", "Smooth Cigar", "Edge-on Disk", "Unbarred Spiral"]


def macro_metrics(y_true, y_pred, probs):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_macro": float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")),
    }


def evaluate_model(model, loader, device=DEVICE):
    model.eval()
    all_targets, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images, return_logits=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    metrics = macro_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs))
    metrics["confusion_matrix"] = confusion_matrix(all_targets, all_preds).tolist()
    return metrics


def run_one(spec, build_model_fn, recipe_name, seed, data, results_dir, weights_dir,
            purpose="run", smoke_test=False, smoke_epochs=3, device=DEVICE, report_every=25):
    """Train and evaluate one (architecture spec, recipe, seed) combination.

    Parameters
    ----------
    spec : dict
        A registry entry (see registry.py). Must contain at least "id" and
        "label"; build_model_fn is responsible for interpreting the rest.
    build_model_fn : callable(spec) -> nn.Module
        Dispatcher for this spec's architecture family (see build.py).
    data : dict
        {"x_train":..., "y_train":..., "x_val":..., "y_val":...,
         "x_test":..., "y_test":...} already-loaded tensors.
    """
    recipe = RECIPES[recipe_name]
    set_all_seeds(seed)
    run_id = f"{spec['id']}__recipe_{recipe_name}__seed_{seed}__{purpose}"
    results_dir = Path(results_dir)
    weights_dir = Path(weights_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{run_id}.json"
    if result_path.exists():
        raise FileExistsError(f"Result already exists: {result_path}. Delete it before intentionally rerunning this job.")

    train_loader, val_loader, test_loader = make_loaders(
        data["x_train"], data["y_train"], data["x_val"], data["y_val"],
        data["x_test"], data["y_test"], recipe_name,
    )
    model = build_model_fn(spec).to(device)
    epochs = smoke_epochs if smoke_test else recipe["epochs"]
    optimizer, scheduler = make_optimizer_and_scheduler(model, recipe_name, epochs_override=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=recipe["label_smoothing"])

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val = -1.0
    best_state = None
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, return_logits=True)
            loss = loss_fn(logits, labels)
            loss.backward()
            if recipe["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"])
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == labels).sum().item())
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_metrics = evaluate_model(model, val_loader, device=device)
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)

        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

        report_step = 1 if smoke_test else report_every
        if (epoch + 1) % report_step == 0 or epoch == epochs - 1:
            print(f"[{run_id}] epoch {epoch+1}/{epochs} train_acc={train_acc:.4f} val_acc={val_metrics['accuracy']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_loader, device=device)
    elapsed = time.time() - t0

    weights_path = weights_dir / f"{run_id}.pt"
    torch.save(model.state_dict(), weights_path)
    result = {
        "run_id": run_id,
        "purpose": purpose,
        "architecture": spec["id"],
        "label": spec.get("label", spec["id"]),
        "family": spec.get("family"),
        "recipe": recipe_name,
        "seed": seed,
        "params": int(sum(p.numel() for p in model.parameters())),
        "reported_params": spec.get("reported_params"),
        "reported_accuracy": spec.get("reported_acc"),
        "reported_f1": spec.get("reported_f1"),
        "train_time_sec": elapsed,
        "epochs": epochs,
        **{k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": test_metrics["confusion_matrix"],
        "history": history,
        "weights_file": str(weights_path.name),
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result
