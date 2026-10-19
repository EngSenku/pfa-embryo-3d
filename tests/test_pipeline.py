#!/usr/bin/env python3
"""Spec -> plan -> parameter injection tests (no network, no notebook execution).

Run:  python3 tests/test_pipeline.py
"""

from __future__ import annotations

import ast
import io
import json
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat as nbf  # noqa: E402
import papermill as pm  # noqa: E402

from pipeline import params_cells  # noqa: E402
from pipeline.spec import Spec, SpecError, build_spec  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def _values(src: str) -> dict:
    vals: dict = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try:
                vals[n.targets[0].id] = ast.literal_eval(n.value)
            except Exception:
                vals[n.targets[0].id] = "<expr>"
    return vals


def _inject(fname: str, params: dict) -> tuple[dict, str]:
    """papermill prepare_only, returning effective values + any papermill log."""
    tmp = Path(tempfile.mkdtemp())
    inp, out = tmp / "in.ipynb", tmp / "out.ipynb"
    nbf.write(nbf.read(str(ROOT / fname), as_version=4), str(inp))
    log = io.StringIO()
    logger = logging.getLogger("papermill")
    handler = logging.StreamHandler(log)
    logger.addHandler(handler)
    try:
        pm.execute_notebook(str(inp), str(out), parameters=params, prepare_only=True)
    finally:
        logger.removeHandler(handler)
    nb = nbf.read(str(out), as_version=4)
    vals: dict = {}
    for cell in nb.cells:
        if set(cell.metadata.get("tags") or []) & {"parameters", "injected-parameters"}:
            vals.update(_values("".join(cell.source)))
    return vals, log.getvalue().strip()


# ---------------------------------------------------------------------------
print("[1] every default is injected into its notebook")
# ---------------------------------------------------------------------------
for fname in params_cells.NOTEBOOK_PARAMS:
    params = params_cells.defaults_for_notebook(fname)
    vals, msgs = _inject(fname, params)
    missing = [k for k in params if k not in vals]
    diff = {k: (v, vals.get(k)) for k, v in params.items()
            if vals.get(k) != v and vals.get(k) != "<expr>"}
    check(fname, not missing and not diff and not msgs,
          f"missing={missing} diff={diff} log={msgs[:200]!r}")

# ---------------------------------------------------------------------------
print("[2] example specs validate and plan")
# ---------------------------------------------------------------------------
plans = {
    "cs17_atlas.yaml": ["registration", "reconstruct"],
    "new_stage_custom_model.yaml": ["registration", "reconstruct"],
    "new_stage_train_from_labels.yaml": ["registration", "data_prep"] +
                                        [f"train_group{g}" for g in range(4)] + ["reconstruct"],
}
for name, expected in plans.items():
    spec = build_spec(ROOT / "specs" / name, run_id="run-test")
    stages = [s.name for s in spec.plan()]
    check(f"{name} plan", stages == expected, f"{stages} != {expected}")

# ---------------------------------------------------------------------------
print("[3] custom-model spec drives MODEL_SOURCE=custom + MODEL_REPO")
# ---------------------------------------------------------------------------
spec = build_spec(ROOT / "specs" / "new_stage_custom_model.yaml", run_id="run-test")
rec = next(s for s in spec.plan() if s.name == "reconstruct")
check("MODEL_SOURCE=custom", rec.params["MODEL_SOURCE"] == "custom",
      rec.params.get("MODEL_SOURCE"))
check("MODEL_REPO", rec.params["MODEL_REPO"] == "my-user/atlas-cs12-nnunet",
      rec.params.get("MODEL_REPO"))
check("registration prunes validation Part A", rec is not None and spec.plan()[0].prune_part_a)

# ---------------------------------------------------------------------------
print("[4] train spec reaches every training notebook with the right group")
# ---------------------------------------------------------------------------
spec = build_spec(ROOT / "specs" / "new_stage_train_from_labels.yaml", run_id="run-test")
trains = {s.name: s for s in spec.plan() if s.name.startswith("train_group")}
for gi in range(4):
    stage = trains.get(f"train_group{gi}")
    check(f"train_group{gi} -> group {gi}",
          stage is not None and stage.params["GROUP_INDEX"] == gi,
          stage.describe() if stage else "missing")
check("train notebooks map to groups",
      {trains[f"train_group{g}"].notebook for g in range(4)} == {
          "trainning0.ipynb", "trainning1.ipynb",
          "trainning2-failed-small-structures.ipynb", "trainning2-3.ipynb"})

# ---------------------------------------------------------------------------
print("[5] invalid specs are rejected")
# ---------------------------------------------------------------------------
for label, raw in [
    ("missing specimen id", {"models": {"source": "pretrained"}}),
    ("bad specimen id", {"specimen": {"id": "has spaces"}, "models": {}}),
    ("bad engine", {"specimen": {"id": "X"}, "run": {"engine": "slurm"}}),
    ("custom without repo", {"specimen": {"id": "X"}, "models": {"source": "custom"}}),
    ("bad input format", {"specimen": {"id": "X"}, "input": {"format": "tiff-stack"}}),
    ("needed not bool", {"specimen": {"id": "X"}, "input": {"needed": "maybe"}}),
]:
    try:
        Spec(raw=raw).validate()
        check(label, False, "was accepted")
    except SpecError as exc:
        check(label, True)
        print(f"         -> {exc}")

# ---------------------------------------------------------------------------
print("[6] all parameter cell texts are papermill-parseable")
# ---------------------------------------------------------------------------
from tools.parameterize_notebooks import unparseable_lines  # noqa: E402

for key, text in params_cells.CELL_SOURCE.items():
    bad = unparseable_lines(text)
    check(key, not bad, str(bad))

# ---------------------------------------------------------------------------
print("[7] Kaggle kernels are addressed by the slug the title resolves to")
# ---------------------------------------------------------------------------
import re as _re  # noqa: E402

from pipeline import runner  # noqa: E402
from pipeline.runner import KaggleEngine, _kernel_id_from_output  # noqa: E402


def _slug(s: str) -> str:
    return _re.sub(r"[^a-z0-9_-]+", "-", s.lower()).strip("-")


check("parses /code/ URL",
      _kernel_id_from_output("... https://www.kaggle.com/code/alice/pfa-run-1-reconstruct ...")
      == "alice/pfa-run-1-reconstruct")
check("parses /notebooks/ URL",
      _kernel_id_from_output("see kaggle.com/notebooks/bob/a-slug") == "bob/a-slug")
check("no URL -> None", _kernel_id_from_output("nothing to see") is None)


class _Proc:
    returncode = 0
    stderr = ""
    stdout = ("Kernel version 1 successfully pushed. Please check progress at "
              "https://www.kaggle.com/code/alice/pfa-run-1-reconstruct")


_captured: dict = {}


def _fake_run(cmd, **kw):
    d = Path(cmd[cmd.index("-p") + 1])
    _captured["meta"] = json.loads((d / "kernel-metadata.json").read_text())
    return _Proc()


_old_run, _old_cmd = runner._run, runner._kaggle_cmd
runner._run, runner._kaggle_cmd = _fake_run, lambda: ["kaggle"]
try:
    with tempfile.TemporaryDirectory() as td:
        nb = Path(td) / "n.ipynb"
        nb.write_text("{}")
        kid = KaggleEngine(username="alice").submit(nb, "pfa-run-1-reconstruct", gpu=False)
finally:
    runner._run, runner._kaggle_cmd = _old_run, _old_cmd

meta = _captured["meta"]
check("submit returns the title-derived id", kid == "alice/pfa-run-1-reconstruct", kid)
check("metadata id matches", meta["id"] == "alice/pfa-run-1-reconstruct", meta.get("id"))
check("title slugifies back to the id slug",
      _slug(meta["title"]) == "pfa-run-1-reconstruct", meta.get("title"))
check("private by default", meta["is_private"] is True)
check("gpu off when the stage does not need it", meta["enable_gpu"] is False)
check("internet on for HF access", meta["enable_internet"] is True)

# Kaggle wins over the id field: if the resolved URL differs, that is what we poll.
_Proc.stdout = "https://www.kaggle.com/code/alice/actually-this-slug"
runner._run, runner._kaggle_cmd = _fake_run, lambda: ["kaggle"]
try:
    with tempfile.TemporaryDirectory() as td:
        nb = Path(td) / "n.ipynb"
        nb.write_text("{}")
        kid = KaggleEngine(username="alice").submit(nb, "pfa-run-1-reconstruct", gpu=True)
finally:
    runner._run, runner._kaggle_cmd = _old_run, _old_cmd
check("resolved URL is the source of truth", kid == "alice/actually-this-slug", kid)
check("gpu on when requested", _captured["meta"]["enable_gpu"] is True)


# --- [8] a fresh Kaggle kernel receives HF_TOKEN from the orchestrator ---
# Kaggle's push API cannot attach account secrets, so the runner injects a
# one-cell preamble.  The value only lands inside the private kernel.
print("\n[8] HF_TOKEN is injected into the pushed kernel")
from pipeline.runner import inject_env  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "n.ipynb"
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_code_cell("x = 1")]
    nbf.write(nb, str(p))
    check("adds a cell when a token is present", inject_env(p, {"HF_TOKEN": "hf_test"}))
    nb2 = nbf.read(str(p), as_version=4)
    first = nb2.cells[0].source
    check("env cell runs first", first.startswith("import os"), first)
    check("sets HF_TOKEN as a literal",
          "os.environ['HF_TOKEN'] = 'hf_test'" in first, first)
    check("keeps the original notebook", any("x = 1" in c.source for c in nb2.cells[1:]))
    check("no empty cell for a missing token",
          not inject_env(p, {"HF_TOKEN": ""})
          and len(nbf.read(str(p), as_version=4).cells) == 2)

# --- [9] resume adopts an in-flight kernel instead of pushing a duplicate ---
# A free host can restart the web tier while a Kaggle kernel is still running.
# Resuming must attach to that kernel (hours of GPU work) rather than submit a
# second copy of the same stage.
print("\n[9] resume adopts an in-flight kernel")
import yaml  # noqa: E402
from pipeline.runner import Runner  # noqa: E402

_raw = yaml.safe_load((ROOT / "specs" / "cs17_atlas.yaml").read_text())
_raw.setdefault("run", {})["mirror_state"] = False   # keep this test offline
_spec = build_spec(_raw, run_id="run-adopt")
_rstage = next(s for s in _spec.plan() if s.name == "reconstruct")

_waited = []
_submitted = []


class _FakeEngine:
    def wait(self, kernel_id):
        _waited.append(kernel_id)

    def fetch_output(self, kernel_id, log_dir):
        _waited.append(("fetched", kernel_id))

    def submit(self, *a, **k):  # pragma: no cover - must never be called
        _submitted.append(a)
        raise AssertionError("resume must not submit a duplicate kernel")


with tempfile.TemporaryDirectory() as _td:
    _r = Runner(spec=_spec, runs_dir=Path(_td), resume=True)
    _r.state = {"run_id": _spec.run_id, "status": "running",
                "stages": {"reconstruct": {"status": "running", "kernel": "u/k"}}}
    _r._engine = lambda: _FakeEngine()  # type: ignore[method-assign]
    _r._run_stage(_rstage)

check("adopts the running kernel", _waited[0] == "u/k", _waited)
check("fetches the adopted kernel's output", ("fetched", "u/k") in _waited, _waited)
check("never submits a duplicate", not _submitted, _submitted)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
