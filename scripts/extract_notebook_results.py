#!/usr/bin/env python3
"""
Dump every code cell's source and text output from the notebooks in
notebooks/ into results/notebook_dumps/*.txt, for anyone who wants to
re-verify a number in results/compiled_training_runs.xlsx (or
scripts/build_report_excel.py) against the notebook it came from, or to
pull out a new number when extending the compiled results.

This is a provenance/verification tool, not a fully automatic table
extractor -- the 15 notebooks in this project print results in enough
different formats (bare print(), pandas display(), custom "===" block
formatting) that a single regex parser silently mis-extracting a value is
a worse outcome than a human eyeballing a grep'd dump. If you're adding a
new architecture's results to the workbook, dump the notebook with this
script, grep the dump for the number you need, and add it to the relevant
list in scripts/build_report_excel.py with a Source pointing at the
notebook filename -- the same process used to build every row already in
results/compiled_training_runs.xlsx.

Usage:
    pip install nbformat
    python scripts/extract_notebook_results.py
    grep -n "Test accuracy" results/notebook_dumps/notebook-best-s4d-model\\(1\\).txt
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
OUT_DIR = REPO_ROOT / "results" / "notebook_dumps"


def get_text_outputs(cell):
    texts = []
    for out in cell.get("outputs", []):
        ot = out.get("output_type")
        if ot == "stream":
            texts.append(out.get("text", ""))
        elif ot in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data:
                t = data["text/plain"]
                texts.append("".join(t) if isinstance(t, list) else t)
        elif ot == "error":
            texts.append("ERROR: " + out.get("ename", "") + ": " + out.get("evalue", ""))
    return "\n".join(texts)


def dump_notebook(nb_path, out_path):
    import nbformat

    nb = nbformat.read(nb_path, as_version=4)
    with open(out_path, "w") as out:
        for i, c in enumerate(nb.cells):
            if c.cell_type == "markdown":
                out.write(f"\n=== [MD {i}] ===\n{c.source}\n")
            elif c.cell_type == "code":
                out.write(f"\n=== [CODE {i}] ===\n{c.source}\n")
                txt = get_text_outputs(c)
                if txt.strip():
                    out.write("--- OUTPUT ---\n" + txt + "\n")


def main():
    try:
        import nbformat  # noqa: F401
    except ImportError:
        print("Missing dependency: pip install nbformat", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found in {NOTEBOOKS_DIR}", file=sys.stderr)
        return 1

    for nb_path in notebooks:
        out_path = OUT_DIR / (nb_path.stem + ".txt")
        dump_notebook(nb_path, out_path)
        print(f"{nb_path.name:55s} -> {out_path.relative_to(REPO_ROOT)} ({out_path.stat().st_size:,} bytes)")

    print(f"\n{len(notebooks)} notebooks dumped to {OUT_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
