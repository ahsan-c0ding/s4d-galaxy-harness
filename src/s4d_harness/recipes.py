"""
The two training recipes used across this project, and the helpers that
build an optimizer/scheduler/seeded-dataloader from them.

Adapted directly from cells 25/27 of notebooks/report-discrepency-testing.ipynb
(the "Controlled Training-Recipe Study" notebook -- the one that discovered
recipe was itself a large confound, Section "Training Recipe as a Confound"
in the LaTeX report). Renamed "addendum" -> "short" to match the report's
own terminology ("main recipe" / "short recipe"); nothing else changed.

This module is the actual mechanism the report used to test six architectures
under both recipes and find an 8-10pp swing in one direction, 2.25-2.55pp in
the other -- so it is deliberately kept as one shared place, rather than
copy-pasted per-notebook the way it was before this harness existed.
"""
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

MAIN_EPOCHS = 630
SHORT_EPOCHS = 40

RECIPES = {
    "main": {
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "epochs": MAIN_EPOCHS,
        "label_smoothing": 0.0,
        "grad_clip": 1.0,
        "scheduler": "warm_restarts",
    },
    "short": {
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": SHORT_EPOCHS,
        "label_smoothing": 0.05,
        "grad_clip": None,
        "scheduler": "cosine",
    },
}


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_augmented_dataset(X, y):
    """Random 90-degree rotation + h/v flip augmentation, train split only."""

    class AugmentedGalaxyDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            img = self.X[idx]
            label = self.y[idx]
            k = random.randint(0, 3)
            if k > 0:
                img = torch.rot90(img, k, dims=(1, 2))
            if random.random() < 0.5:
                img = torch.flip(img, dims=(2,))
            if random.random() < 0.5:
                img = torch.flip(img, dims=(1,))
            return img, label

    return AugmentedGalaxyDataset(X, y)


def make_main_optimizer(model, lr=1e-3, weight_decay=1e-2):
    """AdamW with parameter groups: no weight decay on bias/norm terms or on
    S4D's specially-registered SSM-core parameters (log_A_real, A_imag, ...);
    weight decay everywhere else."""
    decay_params, no_decay_params, special_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if hasattr(param, "_optim"):
            special_params.append({
                "params": [param],
                "lr": getattr(param, "_optim", {}).get("lr", lr),
                "weight_decay": 0.0,
            })
        elif any(k in name for k in ["bias", "norm", "LayerNorm"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups = [
        {"params": decay_params, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
    ] + special_params
    return torch.optim.AdamW(groups)


def make_optimizer_and_scheduler(model, recipe_name, epochs_override=None):
    r = RECIPES[recipe_name]
    epochs = epochs_override or r["epochs"]
    if recipe_name == "main":
        opt = make_main_optimizer(model, lr=r["lr"], weight_decay=r["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=r["lr"], weight_decay=r["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    return opt, sched


def make_loaders(x_train, y_train, x_val, y_val, x_test, y_test, recipe_name):
    recipe = RECIPES[recipe_name]
    train_ds = make_augmented_dataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    test_ds = TensorDataset(x_test, y_test)
    return (
        DataLoader(train_ds, batch_size=recipe["batch_size"], shuffle=True),
        DataLoader(val_ds, batch_size=recipe["batch_size"], shuffle=False),
        DataLoader(test_ds, batch_size=64, shuffle=False),
    )
