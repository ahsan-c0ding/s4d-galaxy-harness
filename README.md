# S4D Galaxy Classifier -- Unified Training/Testing Harness

Team 0x43. Consolidates 15 Kaggle/Colab notebooks' worth of duplicated model
and training code into one tested, importable harness, plus a compiled
Excel workbook of every training run behind the LaTeX report.

## Why this exists

The project's own report (`Isolating the Source of Performance in an S4D
Galaxy Classifier`) found that **training recipe was an undetected
confound for months**: the same architecture scored 8-10 percentage points
differently depending on which notebook's copy-pasted training loop trained
it. That happened because there were 15 near-duplicate notebooks, each
with its own training loop, optimizer setup, and (sometimes subtly
different) model code.

This repo doesn't rewrite any of that code from scratch -- everything here
was extracted verbatim from the notebooks (see **Provenance** below) -- but
it puts it all behind one registry, one model dispatcher, one set of
training recipes, and one train/eval loop, so that class of bug is
structurally harder to reintroduce.

## Layout

```
model/                  Original shared richer-stem package (CNNStem, S4D,
                         GalaxyClassifierS4D/CNNS4D/CNNOnly, HilbertScan, ...),
                         copied verbatim -- byte-identical across 6 of the
                         original notebooks. Untouched on purpose: everything
                         else imports from here rather than redefining it.
utils.py, kaggle_extras.py   Verbatim helper modules from the same package.

src/s4d_harness/
    production_model.py     ConvPatchStem / S4DConv / GalaxyClassifierS4DFast
                             -- the architecture behind the project's headline
                             86.80% result. Verbatim from
                             notebooks/notebook-best-s4d-model(1).ipynb,
                             diffed byte-identical against 8 of the 9
                             single-run production notebooks.
    richer_grid_models.py    Both implementations of the stem-depth x
                             S4D-layer grid (original short-recipe version +
                             main-recipe reconstruction) -- see the module
                             docstring for exactly which table each produced.
    recipes.py               The two training recipes ("main" 630 epochs,
                             "short" 40 epochs) and their optimizer/scheduler
                             builders, extracted from the controlled
                             recipe-crossing study.
    engine.py                One shared train/evaluate loop every
                             architecture now goes through.
    registry.py              Every architecture configuration this project
                             has trained, as data: constructor kwargs +
                             expected parameter count + reported accuracy.
    build.py                 build_model(spec) -- dispatches a registry
                             entry to the right architecture family.

notebooks/               All 15 original notebooks, kept for provenance.
scripts/
    run_experiment.py        CLI: train+eval one registry spec end-to-end.
    build_report_excel.py    Rebuilds results/compiled_training_runs.xlsx.
    extract_notebook_results.py   Dumps notebook cells/outputs to text, for
                             verifying any number against its source notebook.
    legacy/                  The 5 original standalone Kaggle training
                             scripts, kept verbatim for reference.

tests/                    pytest suite -- see below.
results/
    compiled_training_runs.xlsx   Every training run, compiled (see below).
    runs/, weights/                Where new run_experiment.py output lands.
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/src"   # so `model` and `s4d_harness` both import
```

## Running the tests

```bash
pytest -q
```

84 tests, all passing as of this commit. For every one of the 41 registered
architecture configurations, this:

1. Builds the model and asserts its parameter count matches the exact
   figure published in the LaTeX report (`test_registry_param_counts.py`).
2. Runs a forward pass on a random batch and checks the output shape and
   finiteness (`test_forward_pass_shapes.py`).

Plus two regression tests (`test_pooling_last_vs_mean.py`) guarding against
a real bug the report documents finding: `TakeLastTimestep` was once
silently mean-pooling instead of taking the last timestep.

These tests are fast and CPU-only -- no dataset or GPU required, so they're
cheap to run before trusting any future change.

## Running a new experiment

```bash
python scripts/run_experiment.py --list                       # see every registered spec
python scripts/run_experiment.py --id prod_best_s4d_d256 --recipe main --seed 30485
```

This downloads GalaxyMNIST (via the same `galaxy_mnist` package every
notebook used), trains under the requested recipe, evaluates on the test
set, and writes a standardized JSON result to `results/runs/`. Add a new
architecture by adding one entry to `registry.py` -- the training loop,
recipes, and evaluation code don't need to change.

`--smoke-test` runs 3 epochs instead of the full recipe, to sanity-check
the pipeline without waiting for a full run.

## The compiled Excel report

`results/compiled_training_runs.xlsx` -- 9 data sheets + a Summary sheet
with live formulas, covering every training run in the project:

- Production family: master ablation table, recipe-crossing study,
  metric-recovery reruns, repeat-seed follow-up (6 configs x 2 seeds)
- Richer-stem family: CNN-only/S4D-only/hybrid comparison, parameter-scale
  sweep, and the full stem-depth x S4D-layer grid (short + main recipe)
- Noise-robustness sweep (13 models x 5 noise levels) -- raw data that
  isn't tabulated anywhere in the LaTeX report, only discussed qualitatively
- GroupNorm-folding RISC-V portability test

**Provenance**: every row has a `source` column naming the notebook it came
from. Every number was cross-checked against the notebook's actual cell
output, not copied from the report text alone -- spot checks against 10+
notebooks all matched to the last reported decimal. A few cells (noted
individually) trace to the report because the exact producing notebook
cell wasn't independently re-located; those are labeled `LaTeX report`
rather than implied to be notebook-verified.

**Not included**, because it's genuinely not in any notebook: the two
richer-grid cells and the noise-robustness repeat-seed sweep that the
project's own GPU quota ran out before completing (see the report's
"Follow-Up Validation" section).

Rebuild it any time with:
```bash
python scripts/build_report_excel.py
```

## Verifying a number against its source notebook

```bash
python scripts/extract_notebook_results.py
grep -n "Test accuracy" "results/notebook_dumps/notebook-best-s4d-model(1).txt"
```

## What's deliberately NOT unified

`richer_grid_models.py` keeps two separate implementations of the same
grid architecture (original short-recipe version vs. main-recipe
reconstruction) rather than merging them into one, because they are
genuinely two different runs that produced two different tables in the
report. Silently merging them would misrepresent which run produced which
number -- exactly the kind of provenance confusion this harness exists to
prevent.
