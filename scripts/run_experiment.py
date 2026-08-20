#!/usr/bin/env python3
"""
Run one (architecture, recipe, seed) combination through the unified
harness end to end: build the model from registry.py, train it under
recipes.py's main/short recipe, evaluate it, and write a standardized
JSON result to results/runs/.

This replaces the pattern of copy-pasting a training loop into a new
notebook per experiment. To add a new architecture, add one entry to
src/s4d_harness/registry.py (or a new "family" branch in build.py if it's
a genuinely new architecture) -- nothing about the training loop, recipes,
or evaluation needs to be touched.

Usage:
    export PYTHONPATH="$PWD:$PWD/src"
    python scripts/run_experiment.py --id prod_best_s4d_d256 --recipe main --seed 30485
    python scripts/run_experiment.py --list                     # show every registered spec
    python scripts/run_experiment.py --id prod_small_conv_d72 --recipe short --seed 30485 --smoke-test

Data loading is intentionally left to load_galaxymnist() below -- point it
at your own cached GalaxyMNIST tensors, or swap in model.functions.load_data
(the original loader, already vendored into model/) if running somewhere
with network access to download the dataset.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from s4d_harness.build import build_model  # noqa: E402
from s4d_harness.engine import run_one  # noqa: E402
from s4d_harness.registry import ALL_SPECS, get_spec  # noqa: E402


def load_galaxymnist(colored=True, val_fraction=0.2, seed=42):
    """Loads GalaxyMNIST via the vendored model.functions.load_data and
    produces the same train/val/test tensors every notebook in this
    project used. Requires network access to download the dataset the
    first time (see model/functions.py); if you already have cached
    tensors elsewhere, replace this function's body with a loader for
    those instead -- the rest of the harness only cares about the dict
    shape returned below."""
    import torch
    from sklearn.model_selection import train_test_split
    from model.functions import load_data

    x_train_all, _, y_train_all = load_data(root="./data", download=True, train=True, colored=colored)
    x_test, _, y_test = load_data(root="./data", download=True, train=False, colored=colored)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all, y_train_all, test_size=val_fraction, random_state=seed, stratify=y_train_all,
    )
    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", help="registry spec id, e.g. prod_best_s4d_d256")
    parser.add_argument("--recipe", choices=["main", "short"], default="main")
    parser.add_argument("--seed", type=int, default=30485)
    parser.add_argument("--list", action="store_true", help="list every registered spec id and exit")
    parser.add_argument("--smoke-test", action="store_true",
                         help="run only a few epochs to sanity-check the pipeline, don't trust the accuracy")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results" / "runs"))
    parser.add_argument("--weights-dir", default=str(REPO_ROOT / "results" / "weights"))
    args = parser.parse_args()

    if args.list or not args.id:
        for spec in ALL_SPECS:
            print(f"{spec['id']:35s} family={spec['family']:16s} params={spec['expected_params']:>8,}  {spec['label']}")
        if not args.id:
            print("\nPass --id <spec_id> --recipe {main,short} --seed <int> to run one.")
        return 0

    spec = get_spec(args.id)
    data = load_galaxymnist(colored=spec.get("colored", True))
    result = run_one(
        spec=spec, build_model_fn=build_model, recipe_name=args.recipe, seed=args.seed,
        data=data, results_dir=args.results_dir, weights_dir=args.weights_dir,
        purpose="smoke_test" if args.smoke_test else "run", smoke_test=args.smoke_test,
    )
    print(f"\nDone. Test accuracy: {result['accuracy']:.4f}  (report says {spec.get('reported_acc')})")
    print(f"Result written to {args.results_dir}/{result['run_id']}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
