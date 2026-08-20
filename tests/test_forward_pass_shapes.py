"""
Every registered architecture should take a (B, C, 64, 64) GalaxyMNIST-shaped
batch and return (B, 4) logits. This is a cheap smoke test (batch size 2,
random input, no training) that catches shape bugs before they cost a
training run.
"""
import torch
import pytest

from s4d_harness.build import build_model
from s4d_harness.registry import ALL_SPECS

BATCH = 2
NUM_CLASSES = 4


def _input_channels(spec):
    return 3 if spec.get("colored", True) else 1


@pytest.mark.parametrize("spec", ALL_SPECS, ids=[s["id"] for s in ALL_SPECS])
def test_forward_pass_shape(spec):
    model = build_model(spec)
    model.eval()
    x = torch.randn(BATCH, _input_channels(spec), 64, 64)
    with torch.no_grad():
        out = model(x, return_logits=True)
    assert out.shape == (BATCH, NUM_CLASSES), (
        f"{spec['id']}: expected output shape ({BATCH}, {NUM_CLASSES}), got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), f"{spec['id']}: forward pass produced non-finite values"
