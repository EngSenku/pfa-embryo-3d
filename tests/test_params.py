"""Structural tests for the parameterised notebooks.

Run with:  python3 -m pytest tests/ -q   (or python3 tests/test_params.py)

1. every default key really appears as an assignment in its parameters cell;
2. parameters cell + config cell execute together (in a sandbox) for every notebook;
3. papermill injects overrides after the tagged cell (prepare-only);
4. no config cell still assigns a leaf parameter (stomping injection);
5. every hard-coded /kaggle/working root was replaced by {WORK_DIR}.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import params_cells  # noqa: E402

CONFIG_MARKERS = {
    "data-preparation.ipynb": "# 1. CONFIGURATION",
    "trainning0.ipynb": "# 0. CONFIGURATION",
    "trainning1.ipynb": "# 0. CONFIGURATION",
    "trainning2-failed-small-structures.ipynb": "# 0. CONFIGURATION",
    "trainning2-3.ipynb": "# 0. CONFIGURATION",
    "registration-final.ipynb": "USER CONFIGURATION",
    "3d-need-some-work.ipynb": "# 0. CONFIGURATION",
}


def _cells(filename: str):
    nb = nbf.read(str(ROOT / filename), as_version=4)
    pcell = next(c for c in nb.cells
                 if c.cell_type == "code" and "parameters" in (c.metadata.get("tags") or []))
    cfg = next(c for c in nb.cells
               if c.cell_type == "code" and CONFIG_MARKERS[filename] in "".join(c.source))
    return nb, pcell, cfg


def test_defaults_in_cell():
    for fname, pset in params_cells.NOTEBOOK_PARAMS.items():
        text = params_cells.cell_text(pset)
        for key in params_cells.defaults_for_notebook(fname):
            assert f"{key} =" in text, f"{fname}: {key} missing from parameters cell"
    # trainning2-3 uses the composed set
    print("ok - defaults in cell")


def test_cells_execute_together():
    """parameters cell + config cell must run in order without NameError."""
    for fname in params_cells.NOTEBOOK_PARAMS:
        _, pcell, cfg = _cells(fname)
        work = tempfile.mkdtemp(prefix="pfa_work_")
        src = "".join(pcell.source)
        # sandbox the scratch root
        src = src.replace('WORK_DIR = "/kaggle/working"', f'WORK_DIR = {work!r}')
        assert "WORK_DIR =" in src, f"{fname}: parameters cell lost WORK_DIR"
        ns: dict = {"Path": __import__("pathlib").Path}
        exec(compile(src, f"{fname}:parameters", "exec"), ns)
        exec(compile("".join(cfg.source), f"{fname}:config", "exec"), ns)
        assert ns.get("SPECIMEN_ID"), f"{fname}: SPECIMEN_ID not defined"
    print("ok - cells execute together")


def test_config_does_not_stomp():
    import re
    for fname, pset in params_cells.NOTEBOOK_PARAMS.items():
        defaults = params_cells.defaults_for_notebook(fname)
        _, _, cfg = _cells(fname)
        for line in "".join(cfg.source).splitlines():
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=", line)
            if m and m.group(1) in defaults and m.group(1) != "WORK_DIR":
                raise AssertionError(f"{fname}: config cell still assigns {m.group(1)}: {line!r}")
    print("ok - config does not stomp injected params")


def test_no_hardcoded_kaggle_root():
    for fname in params_cells.NOTEBOOK_PARAMS:
        nb, _, _ = _cells(fname)
        for c in nb.cells:
            if c.cell_type != "code":
                continue
            if "parameters" in (c.metadata.get("tags") or []):
                continue
            assert "/kaggle/working" not in "".join(c.source), \
                f"{fname}: hard-coded /kaggle/working remains"
    print("ok - no hard-coded kaggle root")


def test_papermill_injection():
    import papermill as pm

    for fname in params_cells.NOTEBOOK_PARAMS:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out.ipynb"
            pm.execute_notebook(
                str(ROOT / fname), str(out),
                parameters={"SPECIMEN_ID": "CS99-0001"},
                prepare_only=True,
            )
            nb = nbf.read(str(out), as_version=4)
            # papermill inserts the injected cell right after the tagged one
            tags = [i for i, c in enumerate(nb.cells)
                    if c.cell_type == "code" and "parameters" in (c.metadata.get("tags") or [])]
            assert tags, f"{fname}: no parameters cell"
            after = "".join(nb.cells[tags[0] + 1].source)
            assert "SPECIMEN_ID" in after and "CS99-0001" in after, \
                f"{fname}: injection cell missing/incorrect: {after!r}"
    print("ok - papermill injection")


def test_part_a_tagged():
    nb = nbf.read(str(ROOT / "registration-final.ipynb"), as_version=4)
    tagged = [i for i, c in enumerate(nb.cells) if c.metadata.get("pfa_part") == "A"]
    assert len(tagged) >= 8, f"Part A cells not tagged: {tagged}"
    print("ok - registration Part A tagged", tagged)


if __name__ == "__main__":
    test_defaults_in_cell()
    test_cells_execute_together()
    test_config_does_not_stomp()
    test_no_hardcoded_kaggle_root()
    test_papermill_injection()
    test_part_a_tagged()
    print("\nALL OK")
