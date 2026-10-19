"""Run spec: the YAML document a researcher submits from the web form or CLI.

The spec is the single description of *one* pipeline run.  Everything the
notebooks need is derived from it by :class:`Spec`, so a run is fully
reproducible: the spec is saved next to the executed notebooks in
``runs/<run_id>/spec.yaml``.

Sections (all optional unless noted)::

    title: free-text label for the run
    specimen:
      id: REQUIRED. key used for volumes/<id>/, slices/<id>/, viewers/<id>/
      stage: CS17                # Carnegie stage (documentation + Atlas source)
      atlas_specimen_id: "6521"  # Atlas specimen code of the stage
      description: "Carnegie Stage 17, specimen 6521"
    input:                       # registration stage
      needed: auto               # auto | true | false
      format: atlas_am_volume    # atlas_am_volume | raw_section_archive
      atlas_grey_tif_url / atlas_am_url
      raw: {mode, url, archive_url, xy_pixel_size_um, z_spacing_um}
      registration_mode: always  # always | never | auto
      hf_repo / hf_private
    models:
      source: pretrained         # pretrained | custom | train
      training_repo / dataset_repo / groups_manifest_specimen   # pretrained
      model_repo / groups_manifest_in_repo / groups_json / norm  # custom
      dataset_source: atlas      # atlas | nnunet_raw             # train
      groups: [0,1,2,3] / n_groups / trainer / base_dataset_id ...
    reconstruction:
      push_to_hf: true
      max_voxels / fg_threshold / smoke_test_slabs
      mesh: {..any 3D notebook mesh/AO knob..}
      names: {"17": "Heart"}     # optional names.json content
    run:
      engine: kaggle             # kaggle | local
      work_dir: null             # local engine scratch root
      kaggle: {username, gpu, internet, timeout_s}
"""

from __future__ import annotations

import copy
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import params_cells

VALID_ENGINES = ("kaggle", "local")
VALID_MODEL_SOURCES = ("pretrained", "custom", "train")
VALID_INPUT_FORMATS = ("atlas_am_volume", "raw_section_archive")
VALID_DATASET_SOURCES = ("atlas", "nnunet_raw")

NOTEBOOKS = {
    "registration": "registration-final.ipynb",
    "data_prep": "data-preparation.ipynb",
    "train": {0: "trainning0.ipynb", 1: "trainning1.ipynb",
              2: "trainning2-failed-small-structures.ipynb", 3: "trainning2-3.ipynb"},
    "reconstruct": "3d-need-some-work.ipynb",
}


class SpecError(ValueError):
    """Raised when a run spec is invalid."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SpecError(msg)


def _as_dict(value: Any, where: str) -> dict:
    if value is None:
        return {}
    _require(isinstance(value, dict), f"{where} must be a mapping, got {type(value).__name__}")
    return dict(value)


def load_spec(source: str | Path | dict) -> dict:
    """Load a spec from a YAML/JSON path (or accept an already-loaded dict)."""
    if isinstance(source, dict):
        return copy.deepcopy(source)
    p = Path(source)
    _require(p.exists(), f"spec file not found: {p}")
    text = p.read_text()
    data = yaml.safe_load(text) if p.suffix.lower() in (".yaml", ".yml") else None
    if data is None:
        import json
        data = json.loads(text)
    _require(isinstance(data, dict), f"{p}: top level must be a mapping")
    return data


@dataclass
class Stage:
    """One notebook execution inside a run."""

    name: str
    notebook: str
    params: dict
    needs_gpu: bool = False
    gpu: bool = True
    prune_part_a: bool = False
    artifact: dict | None = None   # {kind: hf-file to verify after the run}
    label: str = ""

    def describe(self) -> str:
        return f"{self.name} [{self.notebook}{' GPU' if self.gpu else ''}] {self.label}".rstrip()


@dataclass
class Spec:
    raw: dict
    run_id: str = field(default_factory=lambda: "run-" + secrets.token_hex(4))
    errors: list[str] = field(default_factory=list)

    # --- accessors ---------------------------------------------------------
    @property
    def title(self) -> str:
        return str(self.raw.get("title") or self.specimen.get("id", self.run_id))

    @property
    def specimen(self) -> dict:
        return _as_dict(self.raw.get("specimen"), "specimen")

    @property
    def specimen_id(self) -> str:
        sid = str(self.specimen.get("id") or "").strip()
        _require(bool(sid), "specimen.id is REQUIRED (it becomes volumes/<id>/, viewers/<id>/)")
        _require(re.fullmatch(r"[A-Za-z0-9._-]{1,64}", sid) is not None,
                 f"specimen.id {sid!r} must be letters/digits/dot/underscore/hyphen only")
        return sid

    @property
    def stage(self) -> str:
        return str(self.specimen.get("stage") or "CS17")

    @property
    def atlas_specimen_id(self) -> str:
        return str(self.specimen.get("atlas_specimen_id") or "")

    @property
    def input(self) -> dict:
        return _as_dict(self.raw.get("input"), "input")

    @property
    def models(self) -> dict:
        return _as_dict(self.raw.get("models"), "models")

    @property
    def reconstruction(self) -> dict:
        return _as_dict(self.raw.get("reconstruction"), "reconstruction")

    @property
    def run_cfg(self) -> dict:
        return _as_dict(self.raw.get("run"), "run")

    @property
    def engine(self) -> str:
        eng = str(self.run_cfg.get("engine") or "kaggle")
        _require(eng in VALID_ENGINES, f"run.engine must be one of {VALID_ENGINES}, got {eng!r}")
        return eng

    @property
    def shared_repo(self) -> str:
        return str(self.input.get("hf_repo") or "senku111/embryo-3d")

    @property
    def model_source(self) -> str:
        src = str(self.models.get("source") or "pretrained")
        _require(src in VALID_MODEL_SOURCES,
                 f"models.source must be one of {VALID_MODEL_SOURCES}, got {src!r}")
        return src

    # --- validation --------------------------------------------------------
    def validate(self) -> None:
        """Raise :class:`SpecError` on the first problem found."""
        _ = self.specimen_id
        self._validate_input()
        self._validate_models()
        _ = self.engine

    def _validate_input(self) -> None:
        fmt = str(self.input.get("format") or "atlas_am_volume")
        _require(fmt in VALID_INPUT_FORMATS,
                 f"input.format must be one of {VALID_INPUT_FORMATS}, got {fmt!r}")
        needed = self.input.get("needed", "auto")
        if not isinstance(needed, bool):
            needed = str(needed)
        _require(needed in ("auto", "true", "false", True, False),
                 f"input.needed must be auto/true/false, got {needed!r}")
        if fmt == "atlas_am_volume":
            url = self.input.get("atlas_grey_tif_url")
            if url is None and not self.atlas_specimen_id:
                raise SpecError("input.format=atlas_am_volume needs input.atlas_grey_tif_url "
                                "or specimen.atlas_specimen_id")
        else:
            raw = _as_dict(self.input.get("raw"), "input.raw")
            mode = str(raw.get("mode") or "url")
            _require(mode in ("url", "local_archive"), f"input.raw.mode must be url/local_archive")
            if needed in (True, "true") and mode == "url":
                _require(bool(raw.get("url") or raw.get("archive_url")),
                         "input.raw.url is required when input.format=raw_section_archive")
            xy, z = raw.get("xy_pixel_size_um"), raw.get("z_spacing_um")
            if xy is None or z is None:
                raise SpecError(
                    "raw serial sections need a physical scale: input.raw.xy_pixel_size_um and "
                    "input.raw.z_spacing_um (in µm) are required")

    def _validate_models(self) -> None:
        src = self.model_source
        if src == "custom":
            has_inline = bool(self.models.get("groups_json") or self.models.get("groups_manifest"))
            _require(bool(self.models.get("model_repo")) or has_inline,
                     "models.source=custom needs models.model_repo (HF model repo with nnU-Net v2 "
                     "checkpoints + a groups manifest), or an inline models.groups + models.norm")
            if self.models.get("norm"):
                norm = _as_dict(self.models.get("norm"), "models.norm")
                _require("lo" in norm and "hi" in norm, "models.norm needs lo and hi "
                         "(the grey-normalisation range used for training)")
        elif src == "train":
            dsrc = str(self.models.get("dataset_source") or "atlas")
            _require(dsrc in VALID_DATASET_SOURCES,
                     f"models.dataset_source must be one of {VALID_DATASET_SOURCES}, got {dsrc!r}")
            groups = self.models.get("groups")
            if groups is not None:
                _require(isinstance(groups, list) and all(isinstance(g, int) for g in groups),
                         "models.groups must be a list of integers, e.g. [0, 1, 2, 3]")
        # pretrained: the project's own repos are the default; nothing to check

    # --- derived values ----------------------------------------------------
    @property
    def registration_needed(self) -> bool | None:
        """None = decide at run time (auto) by looking for the volume on HF."""
        needed = self.input.get("needed", "auto")
        if needed in (True, "true"):
            return True
        if needed in (False, "false"):
            return False
        return None

    def atlas_urls(self) -> tuple[str, str]:
        """(grey tif url, .am url) for atlas input, honouring spec overrides."""
        if self.input.get("atlas_grey_tif_url"):
            grey = str(self.input["atlas_grey_tif_url"])
        else:
            _require(bool(self.atlas_specimen_id),
                     "input.atlas_grey_tif_url or specimen.atlas_specimen_id required for atlas input")
            base = "http://3datlas.3dembryo.nl"
            grey = f"{base}/3DAtlas_{self.stage}-{self.atlas_specimen_id}-grey-v2016-01.tif"
        am = str(self.input.get("atlas_am_url") or grey.rsplit("-grey-", 1)[0] + "-grey-v2016-01.am")
        return grey, am

    # --- notebook parameter sets -------------------------------------------
    def registration_params(self) -> dict:
        p = params_cells.defaults_for("registration")
        p.update({
            "SPECIMEN_ID": self.specimen_id,
            "SPECIMEN_DESC": self.specimen.get("description")
                or f"{self.stage}, specimen {self.atlas_specimen_id or self.specimen_id}",
            "SPECIMEN_FORMAT": self.input.get("format") or "atlas_am_volume",
            "REGISTRATION_MODE": self.input.get("registration_mode") or "always",
            "PUSH_TO_HF": bool(self.input.get("push_to_hf", True)),
            "HF_REPO_ID": self.shared_repo,
            "HF_PRIVATE": bool(self.input.get("hf_private", False)),
            "USE_NON_RIGID_OUTPUT": bool(self.input.get("use_non_rigid_output", False)),
            "VALIS_MAX_PROCESSED_IMAGE_DIM_PX":
                int(self.input.get("valis_max_dim_px") or p["VALIS_MAX_PROCESSED_IMAGE_DIM_PX"]),
        })
        fmt = p["SPECIMEN_FORMAT"]
        if fmt == "atlas_am_volume":
            grey, am = self.atlas_urls()
            p["ATLAS_GREY_TIF_URL"], p["ATLAS_AM_URL"] = grey, am
        else:
            raw = _as_dict(self.input.get("raw"), "input.raw")
            p["RAW_INPUT_MODE"] = raw.get("mode") or "url"
            p["RAW_ARCHIVE_URL"] = raw.get("url") or raw.get("archive_url")
            p["RAW_LOCAL_PATH"] = raw.get("local_path")
            p["RESEARCHER_SUPPLIED_METADATA"] = {
                "xy_pixel_size_um": raw.get("xy_pixel_size_um"),
                "z_cutting_interval_um": raw.get("z_spacing_um"),
                "units": "um",
            }
        if self.run_cfg.get("work_dir"):
            p["WORK_DIR"] = str(self.run_cfg["work_dir"])
        return p

    def data_prep_params(self) -> dict:
        p = params_cells.defaults_for("data-preparation")
        m = self.models
        p.update({
            "STAGE": self.stage,
            "ATLAS_SPECIMEN_ID": self.atlas_specimen_id or p["ATLAS_SPECIMEN_ID"],
            "SPECIMEN_ID": self.specimen_id,
            "HF_REPO_ID": str(m.get("dataset_repo") or p["HF_REPO_ID"]),
            "N_GROUPS": int(m.get("n_groups") or p["N_GROUPS"]),
            "BASE_DATASET_ID": int(m.get("base_dataset_id") or p["BASE_DATASET_ID"]),
            "DATASET_NAME_PREFIX": str(m.get("dataset_name_prefix") or p["DATASET_NAME_PREFIX"]),
            "GROUPING_STRATEGY": str(m.get("grouping") or p["GROUPING_STRATEGY"]),
            "SPLIT_SEED": int(m.get("split_seed") or p["SPLIT_SEED"]),
            "HOLDOUT_FRACTION": float(m.get("holdout_fraction") or p["HOLDOUT_FRACTION"]),
        })
        if self.run_cfg.get("work_dir"):
            p["WORK_DIR"] = str(self.run_cfg["work_dir"])
        return p

    @property
    def n_groups(self) -> int:
        if self.models.get("n_groups"):
            return int(self.models["n_groups"])
        if self.models.get("groups"):
            return len(self.models["groups"])
        return int(params_cells.DEFAULTS["reconstruction"]["N_GROUPS"])  # default: 4

    @property
    def groups_to_train(self) -> list[int]:
        groups = self.models.get("groups")
        if groups:
            return [int(g) for g in groups]
        return list(range(self.n_groups))

    def training_params(self, group_index: int) -> dict:
        fname = NOTEBOOKS["train"][group_index]
        p = params_cells.defaults_for_notebook(fname)
        m = self.models
        p.update({
            "SPECIMEN_ID": self.specimen_id,
            "GROUP_INDEX": int(group_index),
            "HF_TRAINING_REPO_ID": str(m.get("training_repo") or p["HF_TRAINING_REPO_ID"]),
            "DATASET_HF_REPO_ID": str(m.get("dataset_repo") or p["DATASET_HF_REPO_ID"]),
            "TRAINER": str(m.get("trainer") or p["TRAINER"]),
            "RESUME_FROM_CHECKPOINT": bool(m.get("resume_from_checkpoint", True)),
        })
        if m.get("checkpoint_upload_minutes"):
            p["CHECKPOINT_UPLOAD_MINUTES"] = int(m["checkpoint_upload_minutes"])
        if self.run_cfg.get("work_dir"):
            p["WORK_DIR"] = str(self.run_cfg["work_dir"])
        return p

    def reconstruction_params(self) -> dict:
        p = params_cells.defaults_for("reconstruction")
        m, r = self.models, self.reconstruction
        src = self.model_source
        p.update({
            "STAGE": self.stage,
            "ATLAS_SPECIMEN_ID": self.atlas_specimen_id or p["ATLAS_SPECIMEN_ID"],
            "SPECIMEN_ID": self.specimen_id,
            "HF_SHARED_REPO_ID": self.shared_repo,
            "N_GROUPS": self.n_groups,
            "MODEL_SOURCE": "custom" if src == "custom" else "existing",
            "PUSH_TO_HF": bool(r.get("push_to_hf", True)),
        })
        if src == "custom":
            p["MODEL_REPO"] = m.get("model_repo")
            p["GROUPS_MANIFEST_IN_REPO"] = m.get("groups_manifest_in_repo")
            if m.get("groups_json"):
                gj = m["groups_json"]
                p["GROUPS_JSON"] = gj if isinstance(gj, str) else _groups_manifest_payload(gj, m)
            if m.get("norm"):
                norm = _as_dict(m.get("norm"), "models.norm")
                p["NORM_OVERRIDE_LO"] = float(norm["lo"])
                p["NORM_OVERRIDE_HI"] = float(norm["hi"])
        else:  # pretrained, or checkpoints this run just trained
            p["HF_TRAINING_REPO_ID"] = str(m.get("training_repo") or p["HF_TRAINING_REPO_ID"])
            p["HF_DATASET_REPO_ID"] = str(m.get("dataset_repo") or p["HF_DATASET_REPO_ID"])
            p["GROUPS_MANIFEST_SPECIMEN_ID"] = m.get("groups_manifest_specimen") or self.specimen_id
        # generic overrides
        for key in ("max_voxels", "fg_threshold", "checkpoint_policy", "disable_tta",
                    "infer_all_slices", "smoke_test_slabs"):
            if key in r:
                p[key.upper()] = r[key]
        if r.get("checkpoint_override") is not None:
            p["CHECKPOINT_OVERRIDE"] = {
                (int(k) if str(k).isdigit() else k): v
                for k, v in _as_dict(r.get("checkpoint_override"), "reconstruction.checkpoint_override").items()
            }
        mesh = _as_dict(r.get("mesh"), "reconstruction.mesh")
        for key, val in mesh.items():
            ukey = key.upper()
            _require(ukey in p, f"reconstruction.mesh.{key} is not a 3D notebook parameter")
            p[ukey] = type(p[ukey])(val) if p[ukey] is not None else val
        if self.run_cfg.get("work_dir"):
            p["WORK_DIR"] = str(self.run_cfg["work_dir"])
        return p

    # --- plan --------------------------------------------------------------
    def plan(self) -> list[Stage]:
        """Ordered list of notebook executions for this spec."""
        stages: list[Stage] = []
        src = self.model_source

        stages.append(Stage(
            name="registration",
            notebook=NOTEBOOKS["registration"],
            params=self.registration_params(),
            gpu=False,
            prune_part_a=True,
            artifact={"kind": "volume", "specimen": self.specimen_id,
                      "repo": self.shared_repo},
            label=f"register {self.specimen_id}",
        ))

        if src == "train" and str(self.models.get("dataset_source") or "atlas") == "atlas":
            stages.append(Stage(
                name="data_prep",
                notebook=NOTEBOOKS["data_prep"],
                params=self.data_prep_params(),
                gpu=False,
                artifact={"kind": "groups_manifest", "specimen": self.specimen_id,
                          "repo": str(self.models.get("dataset_repo")
                                      or params_cells.DEFAULTS["data-preparation"]["HF_REPO_ID"])},
                label=f"{self.n_groups} groups for {self.specimen_id}",
            ))

        if src == "train":
            for gi in self.groups_to_train:
                stages.append(Stage(
                    name=f"train_group{gi}",
                    notebook=NOTEBOOKS["train"][gi],
                    params=self.training_params(gi),
                    gpu=True,
                    label=f"train group {gi} ({self.n_groups} groups)",
                ))

        stages.append(Stage(
            name="reconstruct",
            notebook=NOTEBOOKS["reconstruct"],
            params=self.reconstruction_params(),
            gpu=True,
            artifact={"kind": "viewer", "specimen": self.specimen_id,
                      "repo": self.shared_repo},
            label=f"3D bundle for {self.specimen_id}",
        ))
        return stages


def _groups_manifest_payload(groups: Any, models: dict) -> dict:
    """Build a full groups manifest from a spec's inline `models.groups`."""
    _require(isinstance(groups, list) and groups, "models.groups must be a non-empty list")
    norm = _as_dict(models.get("norm"), "models.norm")
    _require("lo" in norm and "hi" in norm, "inline models.groups requires models.norm.{lo,hi}")
    out_groups = []
    for i, g in enumerate(groups):
        if isinstance(g, dict):
            entry = {"group_index": int(g.get("group_index", i)),
                     "dataset_id": int(g["dataset_id"]),
                     "dataset_name": str(g["dataset_name"]),
                     "atlas_ids": [int(a) for a in g["atlas_ids"]],
                     "n_classes": int(g.get("n_classes", len(g["atlas_ids"]))),
                     "total_positive_pixels": int(g.get("total_positive_pixels", 0))}
        else:  # shorthand: plain structure ids -> one group
            entry = {"group_index": i, "dataset_id": 701 + i,
                     "dataset_name": f"SpecimenGroup{i:02d}",
                     "atlas_ids": [int(a) for a in g],
                     "n_classes": len(g), "total_positive_pixels": 0}
        out_groups.append(entry)
    return {
        "specimen": models.get("specimen") or "",
        "normalization": {"global_lo": float(norm["lo"]), "global_hi": float(norm["hi"]),
                          "source": "spec"},
        "groups": out_groups,
    }


def build_spec(source: str | Path | dict, run_id: str | None = None) -> Spec:
    """Load + validate a spec and stamp a run id."""
    raw = load_spec(source)
    spec = Spec(raw=raw, run_id=run_id or raw.get("run_id") or "run-" + secrets.token_hex(4))
    spec.validate()
    return spec
