#!/usr/bin/env python3
"""Parameterise the pipeline notebooks (idempotent).

What it does, per notebook:

1. ensures a ``parameters``-tagged cell exists at the top (content from
   ``pipeline/params_cells.py``) so papermill can inject values;
2. removes the now-injected leaf assignments from the config cell, keeping all
   derived values, imports, guards and prints;
3. replaces hard-coded ``/kaggle/working`` scratch roots with ``{WORK_DIR}``;
4. tags the registration notebook's Part A (validation) cells with
   ``pfa_part: A`` so the runner can drop them for production runs.

Originals are backed up once to ``backups/original_notebooks/``.
Run from the project root:

    python3 tools/parameterize_notebooks.py            # apply
    python3 tools/parameterize_notebooks.py --check     # report only
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import uuid
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import params_cells  # noqa: E402

BACKUP_DIR = ROOT / "backups" / "original_notebooks"

# where the config cell lives, per notebook
CONFIG_MARKERS = {
    "data-preparation.ipynb": "# 1. CONFIGURATION",
    "trainning0.ipynb": "# 0. CONFIGURATION",
    "trainning1.ipynb": "# 0. CONFIGURATION",
    "trainning2-failed-small-structures.ipynb": "# 0. CONFIGURATION",
    "trainning2-3.ipynb": "# 0. CONFIGURATION",
    "registration-final.ipynb": "USER CONFIGURATION",
    "3d-need-some-work.ipynb": "# 0. CONFIGURATION",
}

# Part A of the registration notebook is the soutenance validation (independent of
# production). Its cells are tagged so the runner can drop them.
PART_A_CELLS = {
    "registration-final.ipynb": [4, 5, 6, 7, 8, 9, 10, 11, 12],
}

_OPEN = "([{"
_CLOSE = ")]}"


def _balance(line: str) -> int:
    return sum(line.count(c) for c in _OPEN) - sum(line.count(c) for c in _CLOSE)


def strip_assignments(source: str, names: set[str]) -> str:
    """Remove top-level assignments of ``names`` (handles multi-line values).

    Recognises ``NAME = ...``, ``NAME, OTHER = ...`` and annotated forms
    (``NAME: type = ...``), and keeps every other line untouched.
    """
    out: list[str] = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=", line)
        if not m:
            # augmented / tuple target such as `AO_BIAS, AO_GAIN = 0.28, 1.8`
            m2 = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
            if m2 and m2.group(1) in names and m2.group(2) in names:
                print(f"    - removed: {line.strip()}")
                i += 1
                continue
            out.append(line)
            i += 1
            continue
        if m.group(1) in names:
            print(f"    - removed: {line.strip()[:80]}")
            bal = _balance(line)
            i += 1
            while bal > 0 and i < len(lines):
                bal += _balance(lines[i])
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + ("\n" if source.endswith("\n") else "")


def find_config_cell(nb, marker: str):
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and marker in "".join(cell.source):
            return idx, cell
    raise SystemExit(f"config cell not found (marker {marker!r})")


def find_param_cell(nb):
    for cell in nb.cells:
        if cell.cell_type == "code" and "parameters" in (cell.metadata.get("tags") or []):
            return cell
    return None


def unparseable_lines(text: str) -> list[tuple[int, str]]:
    """Lines papermill's parameter translator cannot parse.

    The translator splits the cell on lines containing ``=`` and warns
    ("Unable to parse line ...") when a single line contains more than one,
    which silently drops that parameter.  Trailing comments must therefore
    never contain ``=``.
    """
    bad = []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.count("=") > 1:
            bad.append((i, line))
    return bad


def process(filename: str, check: bool) -> bool:
    path = ROOT / filename
    nb = nbf.read(str(path), as_version=4)
    param_set = params_cells.NOTEBOOK_PARAMS[filename]
    defaults = params_cells.defaults_for_notebook(filename)
    names = set(defaults) - {"WORK_DIR"}
    changed = False

    for lineno, line in unparseable_lines(params_cells.cell_text(param_set)):
        print(f"  ! parameter cell line {lineno} has '=' in a comment and will be "
              f"skipped by papermill: {line.strip()[:70]}")

    # 1. parameters cell -----------------------------------------------------
    text = params_cells.cell_text(param_set)
    cell = find_param_cell(nb)
    if cell is None:
        cell = nbf.v4.new_code_cell(text)
        cell.metadata["tags"] = ["parameters"]
        nb.cells.insert(1, cell)
        print(f"  + inserted parameters cell at index 1")
        changed = True
    elif "".join(cell.source) != text:
        cell.source = text
        print("  ~ refreshed parameters cell")
        changed = True

    # 2. strip injected leaf assignments from the config cell ---------------
    cfg_idx, cfg = find_config_cell(nb, CONFIG_MARKERS[filename])
    new_src = strip_assignments("".join(cfg.source), names)
    if new_src != "".join(cfg.source):
        cfg.source = new_src
        changed = True

    # 3. scratch root --------------------------------------------------------
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        if "parameters" in (c.metadata.get("tags") or []):
            continue  # the parameters cell keeps its documented default
        src = "".join(c.source)
        if "/kaggle/working" in src:
            c.source = src.replace("/kaggle/working", "{WORK_DIR}")
            print("    ~ scratch root -> {WORK_DIR}")
            changed = True

    # 4. tag Part A cells ----------------------------------------------------
    for idx in PART_A_CELLS.get(filename, []):
        if "pfa_part" not in nb.cells[idx].metadata:
            nb.cells[idx].metadata["pfa_part"] = "A"
            changed = True

    if not check and changed:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup = BACKUP_DIR / filename
        if not backup.exists():
            shutil.copy2(path, backup)
        nbf.write(nb, str(path))
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report what would change")
    ap.add_argument("--only", default=None, help="single notebook file name")
    args = ap.parse_args()

    files = [args.only] if args.only else list(params_cells.NOTEBOOK_PARAMS)
    any_change = False
    for f in files:
        print(f"[{f}]")
        if process(f, args.check):
            any_change = True
            print("  => changed" if not args.check else "  => would change")
        else:
            print("  => already up to date")
    if args.check and any_change:
        sys.exit(1)


if __name__ == "__main__":
    main()
