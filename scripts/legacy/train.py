#!/usr/bin/env python
# coding: utf-8
"""
scripts/train.py -- trimmed for the Kaggle ablation-runs notebook.

This keeps the exact `train()` function from the original repo file
unchanged (it's reused, by direct file-path import, by train_hybrid.py,
train_cnn_only.py, and train_hybrid_scale.py). The original file's guarded
`if __name__ == "__main__":` block -- ~250 lines of notebook-style baseline
training, an interactive pygame GUI launch, and a weights-export step --
is not reachable from any of those three scripts (it's behind that guard,
and none of them run this file directly as __main__), so it's omitted here
rather than carried along unused. No logic inside `train()` itself was
changed.
"""

# Standard library
import random

# Numerical / plotting
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

# Classifier
from model import GalaxyClassifierS4D
from model.functions import export_model_parameters, load_data

from utils import set_pbar_style


def train(train_loader, val_loader, model, optimizer, loss_fn, epochs, device, verbose=True, scheduler=None):
    """Train the model and validate after each epoch.

    Parameters:
    -----------
    train_loader : DataLoader
        DataLoader for training data.
    val_loader : DataLoader
        DataLoader for validation data.
    model : nn.Module
        The neural network model to train.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    loss_fn : nn.Module
        Loss function to compute training loss.
    epochs : int
        Number of training epochs.
    device : torch.device
        Device to run the training on (CPU or GPU).
    verbose : bool
        Whether to print per-epoch progress.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler, stepped once per epoch after validation.
        If None (default), learning rate stays constant, matching the
        original baseline training regime.

    Returns:
    --------
    history : dict
        Dictionary containing training loss and validation accuracy history.
    """

    history = {
        "loss": [],
        "val_accuracy": []
    }

    ebar = tqdm(range(epochs), desc="Training Progress", disable=verbose)

    # If verbose, don't show outer pbar
    for epoch in ebar:
        model.train()
        running_loss = 0.0

        # show pbar only if verbose
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", disable=not verbose)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, return_logits=True)
            loss = loss_fn(outputs, torch.argmax(targets, dim=1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.set_postfix({"Batch Loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        history["loss"].append(epoch_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, return_logits=True)

                predicted = torch.argmax(outputs, dim=1)
                target = torch.argmax(targets, dim=1)

                correct += (predicted == target).sum().item()
                total += targets.size(0)

        val_accuracy = correct / total
        history["val_accuracy"].append(val_accuracy)

        if scheduler is not None:
            scheduler.step()

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

        ebar.set_postfix({"Loss": epoch_loss, "Val Acc": val_accuracy})

    return history


if __name__ == "__main__":
    print("scripts/train.py: only `train()` is used by this notebook; the "
          "original interactive/export script body was trimmed (see module "
          "docstring). Import this module to reuse train().")
