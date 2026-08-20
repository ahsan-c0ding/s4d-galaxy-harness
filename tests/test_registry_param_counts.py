"""
The most important test in this repo: for every architecture the project
has ever trained (registry.ALL_SPECS), build it and assert the parameter
count matches what the LaTeX report published. This is the harness's
regression guard against the exact class of bug the project already hit
once (report Section "Reproducibility Checks" / "Training Recipe as a
Confound"): silent architecture drift between notebooks.

Fast and CPU-only -- no dataset, no GPU, no training. Every spec in the
registry gets checked every time this file runs.
"""
import pytest

from s4d_harness.build import build_model
from s4d_harness.registry import ALL_SPECS


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("spec", ALL_SPECS, ids=[s["id"] for s in ALL_SPECS])
def test_param_count_matches_report(spec):
    model = build_model(spec)
    actual = count_params(model)
    expected = spec["expected_params"]
    assert actual == expected, (
        f"{spec['id']} ({spec['label']}): built model has {actual:,} params, "
        f"report says {expected:,}. Either the registry's constructor kwargs "
        f"are wrong or the underlying architecture code has changed."
    )
