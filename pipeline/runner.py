"""Execution engines + the run orchestrator.

Two engines share the same preparation step (drop the registration validation
Part A when asked, inject the spec's parameters with papermill):

* ``local``  – executes the prepared notebook on this machine (papermill);
* ``kaggle`` – pushes it as a private Kaggle kernel (GPU when the stage needs
  one), polls until it finishes, then pulls the kernel output for the log.

Kaggle authentication understands both CLI styles: the modern access token
(``KAGGLE_API_TOKEN`` or ``~/.kaggle/access_token``, username introspected from
the token) and the legacy ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` pair (also read
from ``~/.kaggle/kaggle.json``).

Both engines verify the stage's expected artifact on Hugging Face afterwards,
because HF is the transport between stages.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import nbformat as nbf
import papermill as pm
import yaml

from . import hfio, params_cells
from .spec import Spec, Stage

TERMINAL_OK = {"complete", "completed", "succeeded"}
TERMINAL_BAD = {"failed", "error", "cancelled", "canceled", "dead"}


class RunError(RuntimeError):
    """A stage failed (or its expected artifact is missing)."""


# ---------------------------------------------------------------------------
# notebook preparation (shared)
# ---------------------------------------------------------------------------

def prepare_notebook(src: Path, dest: Path, params: dict, prune_part_a: bool = False) -> Path:
    """Write a parameter-injected copy of ``src`` to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    nb = nbf.read(str(src), as_version=4)
    if prune_part_a:
        keep = [c for c in nb.cells if c.metadata.get("pfa_part") != "A"]
        if len(keep) != len(nb.cells):
            nb.cells = keep
    nbf.write(nb, str(dest))
    out = dest.with_suffix(".prepared.ipynb")
    pm.execute_notebook(str(dest), str(out), parameters=params, prepare_only=True)
    out.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------

def _kaggle_cmd() -> list[str]:
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    try:
        import kaggle  # noqa: F401
        return [sys.executable, "-m", "kaggle"]
    except ImportError as e:
        raise RunError(
            "the Kaggle CLI is not installed (`pip install kaggle`); no Kaggle "
            "credentials can be used without it") from e


def _kaggle_username_from_sdk() -> str | None:
    """Ask the Kaggle SDK which account the configured credentials belong to.

    Modern access tokens (``KGAT_...``, in ``KAGGLE_API_TOKEN`` or
    ``~/.kaggle/access_token``) carry the username, so it does not have to be
    configured separately.  Returns ``None`` when nothing is configured or the
    token cannot be introspected (e.g. offline).
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        return None
    try:
        api = KaggleApi()
        api.authenticate()
    except SystemExit:  # the SDK calls exit(1) when no credentials are usable
        return None
    except Exception:
        return None
    return api.config_values.get("username") or None


def kaggle_username() -> str | None:
    """The account for the configured Kaggle credentials (env or introspection)."""
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    return _kaggle_username_from_sdk()


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # the Kaggle CLI reads KAGGLE_API_TOKEN (or KAGGLE_USERNAME / KAGGLE_KEY)
    # from the environment, or ~/.kaggle/access_token / kaggle.json from disk
    return subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)


# Kaggle derives a kernel's slug from its *title*, not from the id in the
# metadata, and prints the resolved URL on push.  Parse it so polling always
# targets the kernel that was really created.
_KERNEL_URL_RE = re.compile(r"kaggle\.com/(?:code|notebooks)/([^/\s]+)/([A-Za-z0-9_-]+)")


def _kernel_id_from_output(text: str) -> str | None:
    m = _KERNEL_URL_RE.search(text or "")
    return f"{m.group(1)}/{m.group(2)}" if m else None


@dataclass
class KaggleEngine:
    username: str | None = None
    gpu_default: bool = True
    internet: bool = True
    is_private: bool = True
    timeout_s: int = 6 * 3600
    poll_s: int = 30
    log: Callable[[str], None] = lambda s: None

    name = "kaggle"

    def _user(self) -> str:
        u = self.username or kaggle_username()
        if not u:
            raise RunError(
                "no usable Kaggle credentials: set KAGGLE_USERNAME (+ KAGGLE_KEY) "
                "or an access token (KAGGLE_API_TOKEN, or ~/.kaggle/access_token) "
                "so the username can be introspected")
        return u

    def submit(self, notebook: Path, slug_base: str, gpu: bool) -> str:
        slug = re.sub(r"[^a-z0-9_-]+", "-", f"{slug_base}".lower()).strip("-")[:60] or "pfa-run"
        kernel_id = f"{self._user()}/{slug}"
        push_dir = notebook.parent / "kaggle_push"
        push_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(notebook, push_dir / "notebook.ipynb")
        meta = {
            "id": kernel_id,
            # Kaggle slugifies the title to derive the kernel slug; keep it a
            # readable form of the same slug so id and reality agree.
            "title": slug.replace("-", " ").replace("_", " ")[:80],
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": self.is_private,
            "enable_gpu": bool(gpu and self.gpu_default),
            "enable_internet": bool(self.internet),
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        proc = _run(_kaggle_cmd() + ["kernels", "push", "-p", str(push_dir)])
        if proc.returncode != 0:
            raise RunError(f"kaggle kernels push failed:\n{proc.stdout}\n{proc.stderr}")
        resolved = _kernel_id_from_output(proc.stdout + "\n" + proc.stderr)
        if resolved and resolved != kernel_id:
            self.log(f"  kaggle: kernel slug resolved to {resolved}")
            kernel_id = resolved
        self.log(f"  kaggle: submitted {kernel_id}")
        return kernel_id

    def status(self, kernel_id: str) -> str:
        proc = _run(_kaggle_cmd() + ["kernels", "status", kernel_id])
        if proc.returncode != 0:
            raise RunError(f"kaggle kernels status failed: {proc.stderr.strip()}")
        text = (proc.stdout + proc.stderr).strip().lower()
        for s in ("running", "queued", "complete", "failed", "error", "cancelled",
                  "canceled", "saving", "cancelled"):
            if s in text:
                return s
        return text or "unknown"

    def wait(self, kernel_id: str) -> None:
        started = time.time()
        last = None
        while True:
            st = self.status(kernel_id)
            if st != last:
                self.log(f"  kaggle: {kernel_id} -> {st}")
                last = st
            if st in TERMINAL_OK:
                return
            if st in TERMINAL_BAD:
                raise RunError(f"Kaggle kernel {kernel_id} ended with status {st!r}")
            if time.time() - started > self.timeout_s:
                raise RunError(f"Kaggle kernel {kernel_id} timed out after {self.timeout_s}s")
            time.sleep(self.poll_s)

    def fetch_output(self, kernel_id: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        proc = _run(_kaggle_cmd() + ["kernels", "output", kernel_id, "-p", str(dest), "--force"])
        if proc.returncode != 0:
            self.log(f"  kaggle: output download failed ({proc.stderr.strip()[:200]})")


@dataclass
class LocalEngine:
    workdir: Path
    log: Callable[[str], None] = lambda s: None

    name = "local"

    def execute(self, notebook: Path, executed: Path, params: dict | None = None) -> None:
        """Run a notebook locally (params already injected; passed for symmetry)."""
        executed.parent.mkdir(parents=True, exist_ok=True)
        pm.execute_notebook(str(notebook), str(executed), kernel_name="python3",
                            progress_bar=False)


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------

@dataclass
class Runner:
    spec: Spec
    runs_dir: Path
    log: Callable[[str], None] = lambda s: None
    dry_run: bool = False
    resume: bool = False
    state: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.runs_dir) / self.spec.run_id
        self.nb_dir = self.run_dir / "notebooks"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.nb_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        # Resuming an existing run id means "skip whatever already finished".
        if self.state_path.exists():
            self.resume = True
        self.state = self._load_state()
        # Local execution gets a per-run scratch root instead of /kaggle/working.
        # The value must land in the spec *before* plan() builds the parameters.
        if self.spec.engine == "local" and not self.spec.run_cfg.get("work_dir"):
            wd = str(self.run_dir / "work")
            self.spec.raw.setdefault("run", {})["work_dir"] = wd
            Path(wd).mkdir(parents=True, exist_ok=True)

    # -- state ------------------------------------------------------------
    def _load_state(self) -> dict:
        if self.resume and self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"run_id": self.spec.run_id, "status": "pending", "stages": {}}

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str))

    def _stages(self) -> list[Stage]:
        return self.spec.plan()

    # -- helpers ----------------------------------------------------------
    def _engine(self):
        if self.spec.engine == "kaggle":
            k = self.spec.run_cfg.get("kaggle") or {}
            return KaggleEngine(
                username=k.get("username"),
                gpu_default=bool(k.get("gpu", True)),
                internet=bool(k.get("internet", True)),
                is_private=bool(k.get("private", True)),
                timeout_s=int(k.get("timeout_s", 6 * 3600)),
                poll_s=int(k.get("poll_s", 30)),
                log=self.log,
            )
        return LocalEngine(workdir=self.run_dir, log=self.log)

    def _needs_registration(self) -> bool:
        needed = self.spec.registration_needed
        if needed is not None:
            return needed
        try:
            exists = hfio.volume_exists(self.spec.specimen_id, self.spec.shared_repo)
        except Exception as e:  # network/token trouble -> be conservative
            self.log(f"  ! could not check for an existing volume ({e}); running registration")
            return True
        if exists:
            self.log(f"  = volumes/{self.spec.specimen_id} already on HF -> skipping registration")
        return not exists

    def _verify(self, stage: Stage) -> None:
        art = stage.artifact
        if not art:
            return
        push_flag = stage.params.get("PUSH_TO_HF", True)
        if not push_flag:
            self.log(f"  = {stage.name}: PUSH_TO_HF=False, nothing to verify")
            return
        kind, sid, repo = art["kind"], art["specimen"], art["repo"]
        checks = {
            "volume": lambda: hfio.volume_exists(sid, repo),
            "groups_manifest": lambda: hfio.groups_manifest_exists(sid, repo),
            "viewer": lambda: hfio.file_exists(
                repo, f"viewers/{sid}/manifest.json", "dataset"),
        }
        for attempt in range(3):
            try:
                if checks[kind]():
                    self.log(f"  ✓ {stage.name}: {kind} artifact present on HF")
                    return
            except Exception as e:
                self.log(f"  ! {stage.name}: artifact check error ({e})")
                return
            time.sleep(5 * (attempt + 1))
        raise RunError(
            f"stage '{stage.name}' finished but its {kind} artifact is missing on "
            f"{repo} — check the kernel log in runs/{self.spec.run_id}/logs/{stage.name}")

    def _verify_training(self, stage: Stage) -> None:
        """Soft check: at least one checkpoint for this group in the training repo."""
        try:
            ds_repo = stage.params["DATASET_HF_REPO_ID"]
            manifest = hfio.read_json(
                ds_repo,
                f"training_dataset/{stage.params['SPECIMEN_ID']}/reports/all_groups_manifest.json",
                "dataset")
            groups = {g["group_index"]: g for g in manifest["groups"]}
            g = groups[stage.params["GROUP_INDEX"]]
            pre = (f"group{g['group_index']:02d}_{g['dataset_name']}/nnUNet_results/"
                   f"Dataset{g['dataset_id']:03d}_{g['dataset_name']}/fold_0/")
            files = hfio.list_repo(stage.params["HF_TRAINING_REPO_ID"], "model")
            hits = [f for f in files if f.startswith(pre) and "checkpoint" in f]
            if hits:
                self.log(f"  ✓ {stage.name}: {len(hits)} checkpoint file(s) at {pre}")
            else:
                self.log(f"  ! {stage.name}: no checkpoint found at {pre} yet "
                         f"(training may have stopped early)")
        except Exception as e:
            self.log(f"  ! {stage.name}: could not verify checkpoints ({e})")

    # -- main -------------------------------------------------------------
    def _preflight(self) -> None:
        """Checks/uploads that must happen before the notebooks run."""
        if self.dry_run:
            return
        m = self.spec.models
        if (self.spec.model_source == "train"
                and str(m.get("dataset_source") or "atlas") == "nnunet_raw"
                and m.get("groups") and m.get("norm")):
            # researcher already has an nnU-Net raw dataset: publish its group
            # manifest so the training notebook can resolve its groups.
            from .spec import _groups_manifest_payload
            repo = str(m.get("dataset_repo") or params_cells.DEFAULTS["data-preparation"]["HF_REPO_ID"])
            path = f"training_dataset/{self.spec.specimen_id}/reports/all_groups_manifest.json"
            if hfio.file_exists(repo, path, "dataset"):
                self.log(f"  = {repo}/{path} already exists, leaving it untouched")
            else:
                payload = _groups_manifest_payload(m["groups"], m)
                payload["specimen"] = self.spec.specimen_id
                hfio.upload_json(repo, path, payload, "dataset",
                                 commit_message=f"groups manifest for {self.spec.specimen_id}")
                self.log(f"  ↑ uploaded {repo}/{path}")

    def run(self) -> dict:
        self.spec.validate()
        stages = self._stages()
        if self.spec.engine == "kaggle" and not self.dry_run:
            self._engine()._user()  # fail fast on missing Kaggle credentials
        self.state["status"] = "running"
        self.state["title"] = self.spec.title
        self.state["engine"] = self.spec.engine
        self.state["specimen_id"] = self.spec.specimen_id
        self._save_state()

        (self.run_dir / "spec.yaml").write_text(
            yaml.safe_dump(self.spec.raw, sort_keys=False))

        # Shared inputs must be in place before any notebook starts.
        self._preflight()

        for stage in stages:
            if stage.name == "registration" and not self._needs_registration():
                self.state["stages"][stage.name] = {"status": "skipped"}
                self._save_state()
                continue
            prev = self.state["stages"].get(stage.name, {})
            if self.resume and prev.get("status") == "done":
                self.log(f"= {stage.name}: already done (resume), skipping")
                continue
            t0 = time.time()
            self.state["stages"][stage.name] = {"status": "running", "started": time.time()}
            self._save_state()
            try:
                self._run_stage(stage)
                if not self.dry_run:
                    self._verify(stage)
                    if stage.name.startswith("train_group"):
                        self._verify_training(stage)
            except Exception as e:
                self.state["stages"][stage.name].update(
                    {"status": "failed", "error": str(e), "ended": time.time()})
                self.state["status"] = "failed"
                self._save_state()
                raise
            self.state["stages"][stage.name].update(
                {"status": "done", "ended": time.time(), "seconds": round(time.time() - t0, 1)})
            self._save_state()

        self.state["status"] = "done"
        self._save_state()
        self.log(f"✓ run {self.spec.run_id} complete")
        return self.state

    def _run_stage(self, stage: Stage) -> None:
        self.log(f"▶ {stage.describe()}")
        src = Path(__file__).resolve().parents[1] / stage.notebook
        if not src.exists():
            raise RunError(f"notebook not found: {src}")
        prepared = self.nb_dir / f"{stage.name}.ipynb"
        prepare_notebook(src, prepared, stage.params, prune_part_a=stage.prune_part_a)

        if self.dry_run:
            self.log(f"  (dry run) prepared {prepared}")
            return

        if self.spec.engine == "local":
            eng = self._engine()
            out = self.nb_dir / f"{stage.name}.executed.ipynb"
            eng.execute(prepared, out, stage.params)
            return

        eng = self._engine()
        slug = f"pfa-{self.spec.run_id}-{stage.name}"
        kernel_id = eng.submit(prepared, slug, gpu=stage.gpu)
        self.state["stages"][stage.name]["kernel"] = kernel_id
        self._save_state()
        try:
            eng.wait(kernel_id)
        finally:
            log_dir = self.run_dir / "logs" / stage.name
            eng.fetch_output(kernel_id, log_dir)
