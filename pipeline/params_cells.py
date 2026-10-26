"""Canonical parameters for every pipeline notebook.

This module is the single source of truth for:

* the text of the ``parameters`` cell that lives at the top of each notebook
  (written by ``tools/parameterize_notebooks.py``), and
* the default values the runner uses when a run spec omits a field
  (used by ``pipeline/spec.py``).

Keeping both here means the notebooks and the runner can never disagree about a
parameter name or its default; ``tests/test_params.py`` asserts that every
default key really appears in the corresponding cell text.
"""

from __future__ import annotations

# --- defaults, per notebook -------------------------------------------------
# notebook key -> ordered {PARAM: default value}

DEFAULTS: dict[str, dict] = {
    "data-preparation": {
        "WORK_DIR": "/kaggle/working",
        "STAGE": "CS17",
        "ATLAS_SPECIMEN_ID": "6521",
        "SPECIMEN_ID": "CS17-6521",
        "ATLAS_BASE_URLS": ["http://3datlas.3dembryo.nl", "https://3datlas.3dembryo.nl"],
        "ATLAS_VERSION_CANDIDATES": ["v2016-01", "v2016-02", "v2016-03"],
        "HF_REPO_ID": "senku111/atlas-cs17-4groups-dataset",
        "HF_REPO_TYPE": "dataset",
        "BASE_DATASET_ID": 701,
        "DATASET_NAME_PREFIX": "AtlasCS17FourGroup",
        "N_GROUPS": 4,
        "GROUPING_STRATEGY": "balanced_contiguous",
        "COUNT_TOLERANCE": 3,
        "COUNT_PENALTY": 0.15,
        "SPLIT_SEED": 42,
        "HOLDOUT_FRACTION": 0.20,
    },
    "training": {
        "WORK_DIR": "/kaggle/working",
        "HF_TRAINING_REPO_ID": "senku111/atlas-cs17-4groups-nnunet-training",
        "DATASET_HF_REPO_ID": "senku111/atlas-cs17-4groups-dataset",
        "SPECIMEN_ID": "CS17-6521",
        "GROUP_INDEX": 0,
        "N_FOLDS": 1,
        "CONFIGURATION": "2d",
        "TRAINER": "nnUNetTrainer_250epochs",
        "CHECKPOINT_UPLOAD_MINUTES": 30,
        "RESUME_FROM_CHECKPOINT": True,
        "CLEAN_RESULTS": False,
    },
    "training_custom_plans": {  # trainning2-3.ipynb only
        "PATCH_SIZE_OVERRIDE": [512, 512],
        "BATCH_SIZE_OVERRIDE": 10,
    },
    "registration": {
        "WORK_DIR": "/kaggle/working",
        "SPECIMEN_ID": "CS17-6521",
        "SPECIMEN_DESC": "Carnegie Stage 17, specimen 6521",
        "SPECIMEN_FORMAT": "atlas_am_volume",
        "STAGE": "CS17",
        "ATLAS_SPECIMEN_ID": "6521",
        "ATLAS_BASE_URLS": ["http://3datlas.3dembryo.nl", "https://3datlas.3dembryo.nl"],
        "ATLAS_VERSION_CANDIDATES": ["v2016-01", "v2016-02", "v2016-03"],
        "ATLAS_GREY_TIF_URL": "http://3datlas.3dembryo.nl/3DAtlas_CS17-6521-grey-v2016-01.tif",
        "ATLAS_AM_URL": "http://3datlas.3dembryo.nl/3DAtlas_CS17-6521-grey-v2016-01.am",
        "RAW_INPUT_MODE": "url",
        "RAW_ARCHIVE_URL": None,
        "RAW_LOCAL_PATH": None,
        "RESEARCHER_SUPPLIED_METADATA": {
            "xy_pixel_size_um": None,
            "z_cutting_interval_um": None,
            "units": "um",
        },
        "REGISTRATION_MODE": "always",
        "USE_NON_RIGID_OUTPUT": False,
        "VALIS_MAX_PROCESSED_IMAGE_DIM_PX": 1024,
        "ALLOW_SECTION_NUMBER_GAPS": False,
        "PUSH_TO_HF": True,
        "HF_REPO_ID": "senku111/embryo-3d",
        "HF_PRIVATE": False,
    },
    "reconstruction": {
        "WORK_DIR": "/kaggle/working",
        "STAGE": "CS17",
        "ATLAS_SPECIMEN_ID": "6521",
        "SPECIMEN_ID": "CS17-6521",
        "HF_SHARED_REPO_ID": "senku111/embryo-3d",
        "HF_DATASET_REPO_ID": "senku111/atlas-cs17-4groups-dataset",
        "HF_TRAINING_REPO_ID": "senku111/atlas-cs17-4groups-nnunet-training",
        "N_GROUPS": 4,
        "CONFIGURATION": "2d",
        "MODEL_FOLD": 0,
        "BUNDLE_TITLE": None,
        "MODEL_SOURCE": "existing",
        "MODEL_REPO": None,
        "GROUPS_MANIFEST_IN_REPO": None,
        "GROUPS_MANIFEST_SPECIMEN_ID": None,
        "GROUPS_JSON": None,
        "NORM_OVERRIDE_LO": None,
        "NORM_OVERRIDE_HI": None,
        "CHECKPOINT_POLICY": "auto",
        "CHECKPOINT_OVERRIDE": {},
        "DISABLE_TTA": True,
        "MAX_VOXELS": 600_000_000,
        "INFER_ALL_SLICES": True,
        "FG_THRESHOLD": 0.40,
        "SMOKE_TEST_SLABS": None,
        "MIN_VOXELS": 40,
        "SMOOTH_MM": 0.014,
        "Z_EXTRA": 1.5,
        "TAUBIN_ITERS": 12,
        "MESH_MAX_CROP": 30_000_000,
        "MAX_FACES": 40_000,
        "MIN_FACES": 1_200,
        "FACES_PER_VOX23": 5.0,
        "LOD0_FRACTION": 0.10,
        "LOD0_MIN_FACES": 150,
        "AO_RADIUS_MM": 0.05,
        "AO_OFFSET_MM": 0.02,
        "AO_BIAS": 0.28,
        "AO_GAIN": 1.8,
        "AO_STRENGTH": 0.6,
        "PUSH_TO_HF": True,
    },
}

# which notebook files use which parameter set
NOTEBOOK_PARAMS: dict[str, str] = {
    "data-preparation.ipynb": "data-preparation",
    "trainning0.ipynb": "training",
    "trainning1.ipynb": "training",
    "trainning2-failed-small-structures.ipynb": "training",
    "trainning2-3.ipynb": "training_custom_plans",
    "registration-final.ipynb": "registration",
    "3d-need-some-work.ipynb": "reconstruction",
}

# ---------------------------------------------------------------------------
# the literal `parameters` cell text written into each notebook
# ---------------------------------------------------------------------------

_HEADER = """\
# ============================================================
# PARAMETERS
# ------------------------------------------------------------
# This cell is tagged `parameters`: the pipeline runner / web form
# injects values for every variable below (papermill), and everything
# after this cell derives its paths and settings from them.
# Running the notebook by hand uses these defaults, which reproduce
# the original CS17 run.
# ============================================================"""

_TAIL = "# ============================================================ # end parameters # ===="

DATA_PREP_CELL = f"""\
{_HEADER}

# Local scratch root. Kaggle: "/kaggle/working"; Colab: "/content".
WORK_DIR = "/kaggle/working"

# --- Atlas input ------------------------------------------------------------
# Every Atlas Carnegie stage uses its own specimen code in its filenames
# (CS15-3512, CS17-6521, CS12-8505A, CS09-H712, ...). The specimen code is the
# part after the stage in "3DAtlas_<STAGE>-<CODE>-grey-v<VER>.tif".
STAGE = "CS17"
ATLAS_SPECIMEN_ID = "6521"
SPECIMEN_ID = "CS17-6521"  # must equal f"{{STAGE}}-{{ATLAS_SPECIMEN_ID}}"
ATLAS_BASE_URLS = ["http://3datlas.3dembryo.nl", "https://3datlas.3dembryo.nl"]
# The Atlas used different filename version suffixes across stages (v2016-01 for
# CS15, v2016-03 for CS09/CS12, ...). Every candidate is probed against the live
# server in section 2 and whichever one exists wins -- no guessing.
ATLAS_VERSION_CANDIDATES = ["v2016-01", "v2016-02", "v2016-03"]

# --- where the training datasets + reports are pushed -----------------------
HF_REPO_ID = "senku111/atlas-cs17-4groups-dataset"
HF_REPO_TYPE = "dataset"
# group 0 -> BASE_DATASET_ID, group 1 -> BASE_DATASET_ID+1, ...
BASE_DATASET_ID = 701
DATASET_NAME_PREFIX = "AtlasCS17FourGroup"

# --- balanced grouping of the structures into N multiclass models -----------
N_GROUPS = 4
GROUPING_STRATEGY = "balanced_contiguous"  # or "snake"
COUNT_TOLERANCE = 3   # classes per group may differ by +/- this many
COUNT_PENALTY = 0.15  # extra cost per class of count spread

# --- fixed-seed train/held-out slice split ----------------------------------
SPLIT_SEED = 42
HOLDOUT_FRACTION = 0.20

{_TAIL}"""

TRAINING_CELL = f"""\
{_HEADER}

# Local scratch root. Kaggle: "/kaggle/working"; Colab: "/content".
WORK_DIR = "/kaggle/working"

# Single shared "training checkpoints" repo (repo_type=model). Each group's model
# is pushed to its own folder inside it, so training group 3 never overwrites
# group 0's checkpoints.
HF_TRAINING_REPO_ID = "senku111/atlas-cs17-4groups-nnunet-training"
# The dataset repo the data-preparation notebook pushed to.
DATASET_HF_REPO_ID = "senku111/atlas-cs17-4groups-dataset"
# Must match the SPECIMEN_ID used by the data-preparation notebook.
SPECIMEN_ID = "CS17-6521"

# *** THE ONE VARIABLE TO CHANGE TO TRAIN A DIFFERENT GROUP ***
# Set GROUP_INDEX to the group this run trains (0, 1, 2, ...), then run the
# notebook top to bottom. DATASET_ID / DATASET_NAME / RESULTS_PATH_IN_REPO are
# resolved from that group's entry of all_groups_manifest.json below.
GROUP_INDEX = 0

N_FOLDS = 1
CONFIGURATION = "2d"
TRAINER = "nnUNetTrainer_250epochs"
# Upload the current training state to HF every N minutes.
CHECKPOINT_UPLOAD_MINUTES = 30
# Resume from checkpoint_latest.pth when a fold already exists; False stops instead.
RESUME_FROM_CHECKPOINT = True
# Must stay False. Delete results manually only to intentionally restart a run.
CLEAN_RESULTS = False

{_TAIL}"""

TRAINING_CUSTOM_PLANS_CELL = f"""\
{_HEADER}

# Local scratch root. Kaggle: "/kaggle/working"; Colab: "/content".
WORK_DIR = "/kaggle/working"

HF_TRAINING_REPO_ID = "senku111/atlas-cs17-4groups-nnunet-training"
DATASET_HF_REPO_ID = "senku111/atlas-cs17-4groups-dataset"
SPECIMEN_ID = "CS17-6521"

# *** THE ONE VARIABLE TO CHANGE TO TRAIN A DIFFERENT GROUP ***
GROUP_INDEX = 3

# --- custom plans: smaller patch + bigger batch ------------------------------
# The default plan is 1536x896 on 1821x1126 slices with ~0.2% foreground, which is
# very wasteful. Setting PATCH_SIZE_OVERRIDE creates a separate "nnUNetPlans_smallpatch"
# plans file (a fresh run) with 10 x 512x512 patches (about the same GPU memory).
PATCH_SIZE_OVERRIDE = [512, 512]
BATCH_SIZE_OVERRIDE = 10

N_FOLDS = 1
CONFIGURATION = "2d"
TRAINER = "nnUNetTrainer_250epochs"
CHECKPOINT_UPLOAD_MINUTES = 15
RESUME_FROM_CHECKPOINT = True
CLEAN_RESULTS = False

{_TAIL}"""

REGISTRATION_CELL = f"""\
{_HEADER}

# Local scratch root. Kaggle: "/kaggle/working"; Colab: "/content".
WORK_DIR = "/kaggle/working"

# ---------------- specimen ----------------
SPECIMEN_ID = "CS17-6521"
SPECIMEN_DESC = "Carnegie Stage 17, specimen 6521"
# "atlas_am_volume"    -> a pre-assembled, already-aligned Atlas grey volume (.tif + .am header)
# "raw_section_archive" -> a researcher's own serial sections (registrations runs first)
SPECIMEN_FORMAT = "atlas_am_volume"

# --- atlas input (SPECIMEN_FORMAT == "atlas_am_volume") ---
# Filenames differ by stage and by Atlas release. If the explicit URL is missing,
# the notebook probes these bases and version suffixes against the live server.
STAGE = "CS17"
ATLAS_SPECIMEN_ID = "6521"
ATLAS_BASE_URLS = ["http://3datlas.3dembryo.nl", "https://3datlas.3dembryo.nl"]
ATLAS_VERSION_CANDIDATES = ["v2016-01", "v2016-02", "v2016-03"]
ATLAS_GREY_TIF_URL = "http://3datlas.3dembryo.nl/3DAtlas_CS17-6521-grey-v2016-01.tif"
ATLAS_AM_URL = "http://3datlas.3dembryo.nl/3DAtlas_CS17-6521-grey-v2016-01.am"

# --- raw serial-section input (SPECIMEN_FORMAT == "raw_section_archive") ---
RAW_INPUT_MODE = "url"       # "url" | "local_archive" | "local_folder"
RAW_ARCHIVE_URL = None       # zip/tar of the sections, or a huggingface.co/... file URL
RAW_LOCAL_PATH = None        # path to an archive/folder already present locally

# Required for raw serial sections (physical scale of the input images).
RESEARCHER_SUPPLIED_METADATA = {{
    "xy_pixel_size_um": None,
    "z_cutting_interval_um": None,
    "units": "um",
}}

# ---------------- registration behaviour ----------------
REGISTRATION_MODE = "always"  # "always" | "never" | "auto"
USE_NON_RIGID_OUTPUT = False
VALIS_MAX_PROCESSED_IMAGE_DIM_PX = 1024
ALLOW_SECTION_NUMBER_GAPS = False

# ---------------- output ----------------
PUSH_TO_HF = True
# ONE shared dataset for the whole project: volumes/<specimen>/ (this notebook)
# and viewers/<specimen>/ (3D notebook). It must be PUBLIC so a website can fetch
# the viewer files; Atlas data is public anyway. For unpublished/private specimens
# point HF_REPO_ID at a different, private dataset instead.
HF_REPO_ID = "senku111/embryo-3d"
HF_PRIVATE = False

{_TAIL}"""

RECONSTRUCTION_CELL = f"""\
{_HEADER}

# Local scratch root. Kaggle: "/kaggle/working"; Colab: "/content".
WORK_DIR = "/kaggle/working"

STAGE = "CS17"
ATLAS_SPECIMEN_ID = "6521"
SPECIMEN_ID = "CS17-6521"

# --- Hugging Face ---
# volumes/<id>/ (input) + viewers/<id>/ (output). Must be PUBLIC for the website.
HF_SHARED_REPO_ID = "senku111/embryo-3d"
# Where the group manifest (atlas ids per group + training normalisation) lives
# when MODEL_SOURCE == "existing".
HF_DATASET_REPO_ID = "senku111/atlas-cs17-4groups-dataset"
# Where the group checkpoints live when MODEL_SOURCE == "existing".
HF_TRAINING_REPO_ID = "senku111/atlas-cs17-4groups-nnunet-training"
N_GROUPS = 4

# --- nnU-Net configuration and fold ---
# Researchers may publish 3d_fullres / 3d_lowres models, and a fold other than 0.
CONFIGURATION = "2d"
MODEL_FOLD = 0

# ============================ SEGMENTATION MODEL ============================
# "existing" -> use HF_TRAINING_REPO_ID (+ HF_DATASET_REPO_ID for the manifest).
#               Works for any specimen of the stage the models were trained on.
# "custom"   -> use MODEL_REPO, a researcher-supplied HF model repo holding
#               nnU-Net v2 checkpoints (+ a groups manifest or GROUPS_JSON below).
MODEL_SOURCE = "existing"
MODEL_REPO = None                    # e.g. "my-user/atlas-cs12-nnunet"
GROUPS_MANIFEST_IN_REPO = None       # custom: path of groups_manifest.json inside MODEL_REPO (None for auto)
# existing: the specimen whose manifest describes the model (default = SPECIMEN_ID).
# Point it at the TRAINING specimen when reconstructing a new specimen of the same stage.
GROUPS_MANIFEST_SPECIMEN_ID = None
# custom fallback: inline groups manifest (same schema as all_groups_manifest.json).
GROUPS_JSON = None
# Optional normalisation override (must match how the model was trained).
NORM_OVERRIDE_LO = None
NORM_OVERRIDE_HI = None
# Title stored in the viewer bundle. None shows the specimen id.
BUNDLE_TITLE = None

# --- checkpoint choice per group ---
CHECKPOINT_POLICY = "auto"       # "auto": final if it exists else best | "final" | "best" | "latest"
CHECKPOINT_OVERRIDE = {{}}         # e.g. {{2: "latest"}}
DISABLE_TTA = True               # True: ~4-8x faster inference; False for the final high-quality run

# --- resolution of the 3D label volume ---
MAX_VOXELS = 600_000_000         # uint8 labels -> 600 MB RAM
INFER_ALL_SLICES = True          # predict every slice and pool z-slabs by coverage
FG_THRESHOLD = 0.40              # a label wins a down-sampled voxel when it covers at least 40% of it
SMOKE_TEST_SLABS = None          # e.g. (400, 460) -> only reconstruct slabs 400..459 (fast end-to-end check)

# --- meshing ---
MIN_VOXELS = 40
SMOOTH_MM = 0.014
Z_EXTRA = 1.5
TAUBIN_ITERS = 12
MESH_MAX_CROP = 30_000_000
MAX_FACES = 40_000
MIN_FACES = 1_200
FACES_PER_VOX23 = 5.0
LOD0_FRACTION = 0.10
LOD0_MIN_FACES = 150

# --- baked ambient occlusion ---
AO_RADIUS_MM = 0.05
AO_OFFSET_MM = 0.02
AO_BIAS = 0.28
AO_GAIN = 1.8
AO_STRENGTH = 0.6                # how strongly the viewer applies the baked AO

PUSH_TO_HF = True

{_TAIL}"""

CELL_SOURCE: dict[str, str] = {
    "data-preparation": DATA_PREP_CELL,
    "training": TRAINING_CELL,
    "training_custom_plans": TRAINING_CUSTOM_PLANS_CELL,
    "registration": REGISTRATION_CELL,
    "reconstruction": RECONSTRUCTION_CELL,
}


def cell_text(param_set: str) -> str:
    """Return the `parameters` cell source for a parameter set key."""
    return CELL_SOURCE[param_set]


def defaults_for(param_set: str) -> dict:
    """Return a copy of the default parameter dict for a parameter set key."""
    return dict(DEFAULTS[param_set])


def defaults_for_notebook(filename: str) -> dict:
    """Defaults for a notebook file, resolving composed parameter sets.

    ``trainning2-3.ipynb`` uses the ``training`` parameters plus the custom
    plans knobs.
    """
    param_set = NOTEBOOK_PARAMS[filename]
    if param_set == "training_custom_plans":
        merged = dict(DEFAULTS["training"])
        merged.update(DEFAULTS["training_custom_plans"])
        return merged
    return dict(DEFAULTS[param_set])
