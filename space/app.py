"""Hugging Face Space: web form -> run spec -> the PFA pipeline on Kaggle.

The Space is a thin web layer around ``pipeline.spec`` (what a run *is*) and
``pipeline.runner`` (how it *executes*).  Everything heavy happens in the
researcher's Kaggle account; the Space only

* validates the form into a run spec,
* pushes prepared notebooks / waits / pulls logs via the Kaggle API,
* polls Hugging Face for the produced artifacts,
* serves the 3D viewer once ``viewers/<id>/`` exists.

Endpoints
---------
``GET  /``                 the run form
``GET  /viewer``           the 3D viewer (``index5.html``; ``?specimen=`` selects)
``GET  /api/health``       which credentials are configured
``GET  /api/specimens``    specimens that already have a viewer bundle
``POST /api/plan``         validate the form and return the planned stages
``POST /api/runs``         start a run (form JSON or ``{"yaml": "<spec>"}``)
``GET  /api/runs``         every run and its status
``GET  /api/runs/{run_id}`` live state + the tail of the run log
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]          # project root (ships in the image)
sys.path.insert(0, str(ROOT))

from pipeline import hfio  # noqa: E402
from pipeline.runner import Runner  # noqa: E402
from pipeline.spec import SpecError, build_spec  # noqa: E402

# HF Spaces give every Space a persistent /data mount on Pro/Team plans.
RUNS_DIR = Path(os.environ.get("PFA_RUNS_DIR") or ("/data/pfa-runs" if Path("/data").is_dir()
                                                   else ROOT / "runs"))
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOG_TAIL_LINES = 400
APP_VERSION = "2026-10-26.1"

app = FastAPI(title="PFA pipeline", docs_url="/api/docs", openapi_url="/api/openapi.json")

_lock = threading.Lock()
_threads: dict[str, threading.Thread] = {}


# ---------------------------------------------------------------------------
# specs from the form
# ---------------------------------------------------------------------------

class FormSpec(BaseModel):
    """The form payload.  Only ``specimen_id`` is required; everything else has
    a default that reproduces the CS17 project run."""

    yaml: str | None = Field(None, description="Raw run spec; when present it wins")
    title: str = ""
    specimen_id: str = Field(..., min_length=1)
    stage: str = "CS17"
    atlas_specimen_id: str = ""
    description: str = ""

    input_format: str = "atlas_am_volume"
    input_needed: str = "auto"                      # auto | true | false
    atlas_grey_tif_url: str = ""
    raw_mode: str = "url"
    raw_url: str = ""
    raw_xy_um: float | None = None
    raw_z_um: float | None = None
    registration_mode: str = "always"
    hf_repo: str = "senku111/embryo-3d"
    hf_private: bool = False

    model_source: str = "pretrained"                # pretrained | custom | train
    training_repo: str = "senku111/atlas-cs17-4groups-nnunet-training"
    dataset_repo: str = "senku111/atlas-cs17-4groups-dataset"
    model_repo: str = ""
    groups_manifest_in_repo: str = ""
    groups_json: str = ""                           # inline manifest (custom), YAML/JSON text
    norm_lo: float | None = None
    norm_hi: float | None = None
    model_configuration: str = "2d"                 # nnU-Net configuration
    model_fold: int = 0                             # nnU-Net fold
    dataset_source: str = "atlas"                   # atlas | nnunet_raw
    groups: str = "0,1,2,3"
    n_groups: int = 4
    dataset_name_prefix: str = ""
    base_dataset_id: int | None = None
    grouping: str = "balanced_contiguous"
    trainer: str = ""
    split_seed: int = 42
    holdout_fraction: float = 0.2

    push_to_hf: bool = True
    fg_threshold: float | None = None
    disable_tta: bool | None = None
    max_voxels: int | None = None
    names: str = ""                                 # optional "17": "Heart" JSON

    engine: str = "kaggle"
    kaggle_gpu: bool = True
    kaggle_internet: bool = True
    timeout_s: int = 21600
    dry_run: bool = False
    run_id: str = Field("", description="Reuse an existing run id (with resume)")


def _groups_list(text: str) -> list[int] | None:
    text = (text or "").strip().strip("[]")
    if not text:
        return None
    out = [int(p) for p in text.replace(";", ",").split(",") if p.strip()]
    return out or None


def _inline_json(text: str):
    """Parse an inline YAML/JSON blob (groups list or mapping)."""
    text = (text or "").strip()
    if not text:
        return None
    return yaml.safe_load(text)


def spec_from_form(f: FormSpec) -> dict:
    if f.yaml and f.yaml.strip():
        raw = yaml.safe_load(f.yaml)
        if not isinstance(raw, dict):
            raise SpecError("the advanced spec must be a YAML mapping")
        return raw

    sid = f.specimen_id.strip()
    stage = f.stage.strip()
    atlas_code = f.atlas_specimen_id.strip()
    if "-" in sid:
        head, tail = sid.split("-", 1)
        stage = stage or head
        atlas_code = atlas_code or tail
    spec: dict = {"title": f.title.strip() or f.description.strip() or sid}
    if f.run_id.strip():
        spec["run_id"] = f.run_id.strip()
    spec["specimen"] = {
        "id": sid,
        "stage": stage or "CS17",
        "atlas_specimen_id": atlas_code,
        "description": f.description.strip(),
    }

    inp: dict = {"needed": f.input_needed, "format": f.input_format,
                 "registration_mode": f.registration_mode,
                 "hf_repo": f.hf_repo.strip() or "senku111/embryo-3d",
                 "hf_private": f.hf_private, "push_to_hf": f.push_to_hf}
    if f.atlas_grey_tif_url.strip():
        inp["atlas_grey_tif_url"] = f.atlas_grey_tif_url.strip()
    if f.input_format == "raw_section_archive":
        inp["raw"] = {"mode": f.raw_mode, "url": f.raw_url.strip(),
                      "xy_pixel_size_um": f.raw_xy_um, "z_spacing_um": f.raw_z_um}
    spec["input"] = inp

    m: dict = {"source": f.model_source,
               "training_repo": f.training_repo.strip(),
               "dataset_repo": f.dataset_repo.strip()}
    inline = _inline_json(f.groups_json)
    if f.model_source == "custom":
        m["model_repo"] = f.model_repo.strip()
        m["configuration"] = f.model_configuration.strip() or "2d"
        m["fold"] = int(f.model_fold)
        if f.groups_manifest_in_repo.strip():
            m["groups_manifest_in_repo"] = f.groups_manifest_in_repo.strip()
        if inline is not None:
            if isinstance(inline, list):
                # [0,1,2,3] is rejected by Spec.validate() with an explanation.
                m["groups"] = inline
            elif isinstance(inline, dict):
                if inline.get("groups") is not None:
                    m["groups"] = inline["groups"]
                if inline.get("specimen"):
                    m["specimen"] = inline["specimen"]
                norm = inline.get("normalization") or inline.get("norm")
                if isinstance(norm, dict):
                    lo = norm.get("global_lo", norm.get("lo"))
                    hi = norm.get("global_hi", norm.get("hi"))
                    if lo is not None and hi is not None:
                        m["norm"] = {"lo": float(lo), "hi": float(hi)}
            else:
                raise SpecError("models.groups inline manifest must be a YAML/JSON list "
                                "of group mappings or a mapping with a groups list")
        if f.norm_lo is not None or f.norm_hi is not None:
            m.setdefault("norm", {}).update({"lo": f.norm_lo, "hi": f.norm_hi})
    elif f.model_source == "train":
        m["dataset_source"] = f.dataset_source
        if f.dataset_source == "nnunet_raw":
            # researcher ships an nnU-Net raw dataset + its grouping directly
            if inline is not None:
                m["groups"] = inline if isinstance(inline, list) else inline.get("groups")
                if isinstance(inline, dict) and inline.get("norm"):
                    m["norm"] = inline["norm"]
            if f.norm_lo is not None or f.norm_hi is not None:
                m.setdefault("norm", {}).update({"lo": f.norm_lo, "hi": f.norm_hi})
            return _finish_spec(spec, inp, m, f)
        groups = _groups_list(f.groups)
        if groups is not None:
            m["groups"] = groups
        m["n_groups"] = f.n_groups
        if f.dataset_name_prefix.strip():
            m["dataset_name_prefix"] = f.dataset_name_prefix.strip()
        if f.base_dataset_id is not None:
            m["base_dataset_id"] = f.base_dataset_id
        m["grouping"] = f.grouping
        m["split_seed"] = f.split_seed
        m["holdout_fraction"] = f.holdout_fraction
        if f.trainer.strip():
            m["trainer"] = f.trainer.strip()
    return _finish_spec(spec, inp, m, f)


def _finish_spec(spec: dict, inp: dict, m: dict, f: FormSpec) -> dict:
    spec["models"] = m

    rec: dict = {"push_to_hf": f.push_to_hf}
    if f.fg_threshold is not None:
        rec["fg_threshold"] = f.fg_threshold
    if f.disable_tta is not None:
        rec["disable_tta"] = f.disable_tta
    if f.max_voxels is not None:
        rec["max_voxels"] = f.max_voxels
    names = _inline_json(f.names)
    if isinstance(names, dict):
        rec["names"] = names
    spec["reconstruction"] = rec

    spec["run"] = {
        "engine": f.engine,
        "kaggle": {"gpu": f.kaggle_gpu, "internet": f.kaggle_internet,
                   "timeout_s": f.timeout_s},
    }
    return spec


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

class RunHandle:
    def __init__(self, run_id: str, log_path: Path):
        self.run_id = run_id
        self.log_path = log_path
        self.started = time.time()

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with _lock:
            with self.log_path.open("a") as fh:
                fh.write(line + "\n")


def _shared_repo() -> str:
    """Artifact repo (``volumes/``, ``viewers/``) used by the specimen picker."""
    return os.environ.get("PFA_SHARED_REPO", "senku111/embryo-3d")


def _state_repo() -> str:
    """Where run state is mirrored; must be writable by ``HF_TOKEN``.

    Kept separate from the artifact repo so a researcher can point the artifacts
    at their own repository while the app still finds run state after a restart.
    """
    return os.environ.get("PFA_STATE_REPO") or _shared_repo()


def _read_state(run_id: str, hub_fallback: bool = True) -> dict | None:
    p = RUNS_DIR / run_id / "state.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {"run_id": run_id, "status": "unknown"}
    # After a free-host restart the disk is gone; state lives on the Hub.
    if hub_fallback:
        try:
            return hfio.read_json(_state_repo(), f"runs/{run_id}/state.json", "dataset")
        except Exception:
            return None
    return None


def _log_tail(run_id: str, n: int = LOG_TAIL_LINES) -> str:
    p = RUNS_DIR / run_id / "run.log"
    if not p.exists():
        return ""
    try:
        lines = p.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def start_run(spec_dict: dict, dry_run: bool = False) -> str:
    spec = build_spec(spec_dict)          # validates here, so the caller gets a 4xx
    run_id = spec.run_id
    log_path = RUNS_DIR / run_id / "run.log"

    def _work() -> None:
        handle = RunHandle(run_id, log_path)
        handle.log(f"run {run_id} ({spec.title}) started, engine={spec.engine}")
        try:
            runner = Runner(spec=spec, runs_dir=RUNS_DIR, log=handle.log, dry_run=dry_run)
            runner.run()
        except Exception as exc:  # noqa: BLE001 - the state file records the failure
            handle.log(f"FAILED: {type(exc).__name__}: {exc}")
            state = _read_state(run_id, hub_fallback=False) or {"run_id": run_id, "stages": {}}
            state.setdefault("status", "failed")
            state["error"] = f"{type(exc).__name__}: {exc}"
            (RUNS_DIR / run_id / "state.json").write_text(
                json.dumps(state, indent=2, default=str))

    # Reserve the directory (Runner creates it) so the id appears immediately.
    (RUNS_DIR / run_id).mkdir(parents=True, exist_ok=True)
    thread = threading.Thread(target=_work, name=f"pfa-{run_id}", daemon=True)
    with _lock:
        _threads[run_id] = thread
    thread.start()
    return run_id


def resume_run(run_id: str) -> bool:
    """(Re)start the worker for an unfinished run.

    Free hosts sleep and drop their disk, which kills the worker thread.  This
    rebuilds the spec from the mirrored state and resumes: finished stages are
    skipped and an in-flight Kaggle kernel is adopted rather than duplicated.
    """
    with _lock:
        t = _threads.get(run_id)
        if t and t.is_alive():
            return False
    state = _read_state(run_id)
    if not state or state.get("status") in ("done", "failed"):
        return False
    raw = state.get("spec")
    if not raw:
        return False

    log_path = RUNS_DIR / run_id / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:  # keep the earlier log so a restart does not erase the history
        hfio.download_file(_state_repo(), f"runs/{run_id}/run.log", log_path, "dataset")
    except Exception:
        pass

    def _work() -> None:
        handle = RunHandle(run_id, log_path)
        handle.log(f"resuming run {run_id} ({state.get('status')})")
        try:
            spec = build_spec(raw, run_id=run_id)
            Runner(spec=spec, runs_dir=RUNS_DIR, log=handle.log, resume=True).run()
        except Exception as exc:  # noqa: BLE001
            handle.log(f"FAILED: {type(exc).__name__}: {exc}")

    thread = threading.Thread(target=_work, name=f"pfa-{run_id}", daemon=True)
    with _lock:
        _threads[run_id] = thread
    thread.start()
    return True


def _reap() -> None:
    """Background loop: pick up unfinished runs (local or on the Hub)."""
    while True:
        try:
            ids = {d.name for d in RUNS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")}
            ids |= set(hfio.list_runs(_state_repo()))
            for run_id in ids:
                if _valid_run_id(run_id):
                    resume_run(run_id)
        except Exception:  # noqa: BLE001 - the reaper must never die
            pass
        time.sleep(300)


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/api/version")
def version() -> dict:
    """Which build is answering (lets a deploy be verified without logs)."""
    return {"version": APP_VERSION,
            "commit": (os.environ.get("RENDER_GIT_COMMIT") or "")[:12]}


@app.get("/api/health")
def health(deep: bool = False) -> dict:
    """Credential report.

    The default call is local-only so host health checks (Render, k8s, …) answer
    instantly.  ``?deep=1`` additionally resolves the Kaggle account over the
    network, which is the full end-to-end credential check.
    """
    from pipeline.runner import _kaggle_cmd, kaggle_username  # noqa: PLC0415
    kaggle_dir = Path.home() / ".kaggle"
    creds = {
        "hf_token": bool(hfio.token()),
        # legacy pair (KAGGLE_USERNAME + KAGGLE_KEY, or ~/.kaggle/kaggle.json)
        "kaggle_username": bool(os.environ.get("KAGGLE_USERNAME")),
        "kaggle_key": bool(os.environ.get("KAGGLE_KEY")),
        "kaggle_json": (kaggle_dir / "kaggle.json").exists(),
        # modern access token (username is introspected from it)
        "kaggle_api_token": bool(os.environ.get("KAGGLE_API_TOKEN")),
        "kaggle_access_token_file": (kaggle_dir / "access_token").exists(),
    }
    try:
        _kaggle_cmd()
        creds["kaggle_cli"] = True
    except Exception as exc:  # noqa: BLE001
        creds["kaggle_cli"] = f"missing ({exc})"
    # Network call: resolves (and thereby validates) the account, if any.
    if deep:
        try:
            creds["kaggle_account"] = kaggle_username() or False
        except Exception:  # noqa: BLE001
            creds["kaggle_account"] = False
    else:
        creds["kaggle_account"] = os.environ.get("KAGGLE_USERNAME") or None
    have_kaggle = bool(
        (creds["kaggle_username"] and creds["kaggle_key"])
        or creds["kaggle_api_token"] or creds["kaggle_access_token_file"]
        or creds["kaggle_json"])
    ok = bool(creds["hf_token"]) and creds["kaggle_cli"] is True and have_kaggle
    return {"ok": ok, "creds": creds, "runs_dir": str(RUNS_DIR)}


@app.get("/api/specimens")
def specimens() -> dict:
    repo = os.environ.get("PFA_SHARED_REPO", "senku111/embryo-3d")
    try:
        items = hfio.list_specimens(repo)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Hugging Face listing failed: {exc}") from exc
    return {"repo": repo, "specimens": items}


@app.post("/api/uploads")
async def upload_archive(file: UploadFile = File(...), private: bool = Form(False)) -> dict:
    """Upload a section archive to the Hub and return a direct HTTPS URL.

    Kaggle kernels cannot read this server's disk, so the form must first put
    the ZIP/TAR somewhere reachable.  The file is streamed to a temporary file,
    uploaded to ``PFA_UPLOAD_REPO`` (the shared artifact repo by default), then
    deleted.  For very large datasets, give a dataset URL instead: it avoids
    this server's RAM/disk entirely.
    """
    if not hfio.token():
        raise HTTPException(503, "HF_TOKEN is not configured on this server; "
                                 "give an HTTPS or Hugging Face dataset link instead")
    name = Path(file.filename or "sections.zip").name
    suffix = "".join(Path(name).suffixes).lower()
    allowed = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2",
               ".tar.xz", ".txz", ".rar")
    if not any(suffix.endswith(ext) for ext in allowed):
        raise HTTPException(415, f"unsupported archive type {suffix or '(none)'}; "
                                 f"use one of {', '.join(allowed)}")
    try:
        max_mb = int(os.environ.get("PFA_MAX_UPLOAD_MB", "2048"))
    except ValueError:
        max_mb = 2048
    max_bytes = max_mb * 1024 * 1024
    repo = os.environ.get("PFA_UPLOAD_REPO") or _shared_repo()
    token_dir = RUNS_DIR / "_uploads" / uuid.uuid4().hex
    token_dir.mkdir(parents=True, exist_ok=True)
    tmp = token_dir / name
    size = 0
    try:
        with tmp.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"archive larger than PFA_MAX_UPLOAD_MB={max_mb}; "
                                             "use a dataset link instead")
                fh.write(chunk)
        if size == 0:
            raise HTTPException(400, "empty archive")
        path = f"uploads/{token_dir.name}/{name}"
        api = hfio.api()
        api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
        api.upload_file(path_or_fileobj=str(tmp), path_in_repo=path, repo_id=repo,
                        repo_type="dataset", commit_message=f"upload {name}")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Hugging Face upload failed: {exc}") from exc
    finally:
        shutil.rmtree(token_dir, ignore_errors=True)
    return {
        "repo": repo,
        "path": path,
        "bytes": size,
        "private": bool(private),
        "url": f"https://huggingface.co/datasets/{repo}/resolve/main/{quote(path)}",
    }


@app.post("/api/plan")
def plan(form: FormSpec) -> dict:
    try:
        spec = build_spec(spec_from_form(form))
    except SpecError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - malformed yaml/json in the advanced box
        raise HTTPException(422, f"could not read the spec: {exc}") from exc
    stages = spec.plan()
    return {"run_id": spec.run_id, "specimen_id": spec.specimen_id,
            "engine": spec.engine, "model_source": spec.model_source,
            "spec": spec.raw,
            "stages": [{"name": s.name, "notebook": s.notebook, "gpu": s.gpu,
                        "label": s.label, "params": s.params} for s in stages]}


@app.post("/api/runs")
def create_run(form: FormSpec) -> dict:
    try:
        raw = spec_from_form(form)
    except SpecError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - malformed yaml/json in the advanced box
        raise HTTPException(422, f"could not read the spec: {exc}") from exc
    try:
        run_id = start_run(raw, dry_run=form.dry_run)
    except SpecError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"run_id": run_id, "status": "running", "url": f"/api/runs/{run_id}"}


@app.get("/api/runs")
def list_runs() -> dict:
    ids = {d.name for d in RUNS_DIR.iterdir()
           if d.is_dir() and not d.name.startswith("_")}
    ids |= set(hfio.list_runs(_state_repo()))
    runs = []
    for run_id in sorted(ids, reverse=True):
        state = _read_state(run_id) or {}
        runs.append({"run_id": run_id, "status": state.get("status", "unknown"),
                     "title": state.get("title", ""), "engine": state.get("engine", ""),
                     "specimen_id": state.get("specimen_id", ""),
                     "stages": state.get("stages", {})})
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
def run_state(run_id: str) -> dict:
    if not _valid_run_id(run_id):
        raise HTTPException(400, "bad run id")
    state = _read_state(run_id)
    if state is None:
        raise HTTPException(404, f"unknown run {run_id}")
    # Free hosts sleep and kill the worker: a poll is what brings the run back.
    if state.get("status") not in ("done", "failed"):
        resume_run(run_id)
    return {"state": state, "log": _log_tail(run_id),
            "viewer": _viewer_for(state)}


def _viewer_for(state: dict) -> dict | None:
    sid = state.get("specimen_id")
    if state.get("status") != "done" or not sid:
        return None
    return {"specimen": sid, "url": f"/viewer/?specimen={sid}"}


def _valid_run_id(run_id: str) -> bool:
    return all(c.isalnum() or c in "-_" for c in run_id) and "/" not in run_id


@app.get("/api/runs/{run_id}/log")
def run_log(run_id: str) -> PlainTextResponse:
    if not _valid_run_id(run_id):
        raise HTTPException(400, "bad run id")
    return PlainTextResponse(_log_tail(run_id, 100_000))


# --- static files -----------------------------------------------------------

def _mount() -> None:
    """Serve the 3D viewer (index5.html) and the run form.

    Registered last on purpose: a catch-all mount at "/" would otherwise shadow
    the /api routes defined above it.
    """
    viewer_dir = RUNS_DIR / "_static" / "viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    src = ROOT / "index5.html"
    if src.exists():
        dst = viewer_dir / "index.html"
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            dst.write_text(src.read_text())
    app.mount("/viewer", StaticFiles(directory=viewer_dir, html=True), name="viewer")
    static = Path(__file__).parent / "static"
    app.mount("/", StaticFiles(directory=static, html=True), name="form")


@app.get("/api")
def api_index() -> dict:
    return {"endpoints": ["/api/version", "/api/health", "/api/specimens", "/api/uploads",
                          "/api/plan", "/api/runs", "/api/runs/{run_id}",
                          "/api/runs/{run_id}/log"]}


_mount()


@app.on_event("startup")
def _start_reaper() -> None:
    """Resume runs interrupted by a free-host restart (see ``_reap``)."""
    threading.Thread(target=_reap, name="pfa-reaper", daemon=True).start()


# Make the module importable as `app` for uvicorn.
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
