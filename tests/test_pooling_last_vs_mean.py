"""
Regression guard for a real bug this project already hit once: the report
documents that model/tlts.py's TakeLastTimestep was "silently mean-pooling
despite being named, documented, and self-tested as last-timestep pooling"
before it was caught and fixed prior to any run reported in the paper.

Adapted directly from cell 18 of notebooks/report-discrepency-testing.ipynb,
which is the actual audit that caught it. Keeping it as a permanent test
means this can't quietly regress again.
"""
import torch

from model.tlts import TakeLastTimestep
from s4d_harness.production_model import TakeLastTimestep as ProductionTakeLastTimestep


def _check(layer):
    x = torch.randn(3, 6, 2)
    out = layer(x)
    matches_last = torch.allclose(x[:, -1, :], out)
    matches_mean = torch.allclose(x.mean(dim=1), out)
    assert matches_last, "TakeLastTimestep should return the last timestep"
    assert not matches_mean, "TakeLastTimestep is silently mean-pooling again (this is the bug the report documents finding and fixing)"


def test_richer_stem_take_last_timestep_is_genuine():
    _check(TakeLastTimestep())


def test_production_take_last_timestep_is_genuine():
    _check(ProductionTakeLastTimestep())
