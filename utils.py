"""
Minimal stand-in for `utils.set_pbar_style`, imported by `scripts/train.py`.

`scripts/train.py` does `from utils import set_pbar_style`, but no `utils.py`
was present in the uploaded s4d.zip. Without this stub, importing
`train.py` -- which `train_hybrid.py`, `train_cnn_only.py`, and
`train_hybrid_scale.py` all do, to reuse its `train()` function -- fails
with `ModuleNotFoundError: No module named 'utils'`.

`set_pbar_style` only ever affected tqdm progress-bar colors in the
original notebook-style script body; it has no effect on training logic,
so this no-op-safe stub is a purely cosmetic substitute.
"""


def set_pbar_style(bar_fill_color="#FFFFFF", text_color="#FFFFFF"):
    """Cosmetic no-op. The original styling implementation wasn't in the
    uploaded zip, so this stub just accepts the same call signature used
    in scripts/train.py without changing tqdm's behavior."""
    return None
