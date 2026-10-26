---
title: HistoWeave — 3D histology pipeline
emoji: 🧵
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
tags:
  - pipeline
  - nnunet
  - 3d
  - segmentation
---

# HistoWeave — 3D histology reconstruction pipeline

HistoWeave reconstructs an explorable 3D anatomical model from serial
histological sections. It registers the sections, segments them with nnU-Net
group models, and publishes `volumes/<id>/`, `slices/<id>/` and `viewers/<id>/`
to Hugging Face, where the web viewer reads them directly.

The pipeline is validated on the human embryo (Carnegie stage 17) and can be
used with any serial histology when a segmentation model trained for those
images is supplied.

Everything a researcher has to fill in is a **run spec**. The form on the web
app, the CLI (`run_pipeline.py`) and the notebooks all consume the same spec, so
any run can be replayed exactly.

HistoWeave is the product name; the historical deployment URL, the repository
slug `pfa-embryo-3d` and the `PFA_*` environment variables are kept for
continuity.

The LaTeX report and its compiled PDF are in [`report_pfa/`](report_pfa/);
[`HF_REPOS.md`](HF_REPOS.md) lists every Hugging Face repository and whether the
deployment still needs it.

---

## The three deployment cases

The web form exposes exactly three cases. Training is **not** offered from the
form: the training notebooks remain in the repository for the report and for
reproducing the project models, but a researcher who needs a new model trains it
in their own Kaggle/HF workspace and supplies the resulting model repo.

| Case | Specimen / input | Segmentation model | What the researcher supplies |
| --- | --- | --- | --- |
| **1. CS17 specimen, project model** | `CS17-6521` or another **CS17** specimen (Atlas `.tif + .am`, or new sections) | the project's CS17 4-group models | nothing (the project repos are the default) |
| **2. Other embryo, researcher model** | another embryo: Atlas code, or any section archive / link | a researcher-supplied nnU-Net v2 model | HF model repo + groups manifest + `configuration` + `fold` |
| **3. Any sections, researcher model** | any tissue or object: ZIP/TAR/RAR or Hugging Face dataset link, plus XY µm/px and Z µm/section | a researcher-supplied nnU-Net v2 model | the archive and the physical scale, plus the same model information as case 2 |

Case 1 is stage-specific: the project models are trained for CS17, so a different
specimen is accepted only if it is also CS17. Do not use a CS12/CS15 specimen
with the project model; use case 2 and a model trained for that stage.

Case 2 is intended for another embryo. Case 3 is the generic upload path for any
tissue or object, including non-embryo images. The backend uses the same raw
section path for both; the distinction is the researcher's intent, because the
pipeline cannot verify that an image is an embryo. Case 3 is the upload shortcut
of case 2: it always uses `input.format: raw_section_archive`.

The pipeline does not assume the input is an embryo. It segments whatever
structures the supplied model predicts; the viewer reads the bundle's structure
names and colours from `names.json`.

### Labels and groups

A custom model is not required to use the Atlas 4-group layout. The
reconstruction notebook discovers the group folders and dataset names from the
manifest and the model repo, so a model can have any number of groups, any
`Dataset7xx` ids and any voxel size. The one hard constraint is that the model
must have been trained with the same normalisation range that is recorded in its
manifest.

Inline `models.groups` must be a full list of group mappings:

```yaml
models:
  groups:
    - group_index: 0
      dataset_id: 811
      dataset_name: AtlasCS12Group0
      atlas_ids: [1, 2, 3]
  norm:
    lo: 0
    hi: 60000
```

The old shorthand `groups: [0, 1, 2, 3]` is **rejected**: it is ambiguous
(group indexes? structure ids? labels including background?) and used to map
background `0` as a structure. If the manifest is already published, set
`models.groups_manifest_in_repo` instead.

---

## Inputs

### Atlas volume (`.tif` + `.am`)

The registration notebook builds the URL from the Carnegie stage and the Atlas
specimen code, for example
`3DAtlas_CS17-6521-grey-v2016-01.tif`. The Atlas used different release
suffixes across stages, so the notebook probes `v2016-01`, `v2016-02` and
`v2016-03` against the live server before downloading. An explicit
`input.atlas_grey_tif_url` / `input.atlas_am_url` overrides the probe.

Tested working with `v2016-01`: `CS17-6521`, `CS12-8505A`, `CS15-3512` and
`CS09-H712`. The probe remains because the Atlas has changed suffixes between
releases; `v2016-02`/`v2016-03` return 404 for these four as of October 2026.

### Raw serial sections

* direct ZIP/TAR/RAR URL, or a Hugging Face dataset/file URL (private repos work
  through the server's `HF_TOKEN`);
* or upload a ZIP/TAR/RAR in the form. The app streams it to the shared HF
  dataset with `POST /api/uploads` and puts the resulting HTTPS URL in the spec;
* **XY pixel size (µm/px)** and **Z section spacing (µm/section) are mandatory**
  for raw sections. The registration notebook converts them into the volume
  manifest; without them the viewer cannot report physical units.

`input.raw.mode: local_archive` remains available for the CLI with
`--engine local`, but the form does not expose it and the Kaggle engine rejects
it: a Kaggle kernel cannot see this server's disk.

---

## Models

The project models live in:

| Purpose | Default repo |
| --- | --- |
| 4-group nnU-Net checkpoints | `senku111/atlas-cs17-4groups-nnunet-training` |
| canonical training dataset + `all_groups_manifest.json` (case 1) | `senku111/atlas-cs17-4groups-dataset` |
| shared artifacts (`volumes/`, `slices/`, `viewers/`, run state, uploads) | `senku111/embryo-3d` |

If the training-dataset repo is private or unavailable, case 1 still runs: the
reconstruction notebook derives the same group layout from the `dataset.json`
files stored next to the published checkpoints (label names carry the Atlas
ids), then uses the CS17 normalisation range. This is a read-only fallback; no
extra upload is required.

The Hugging Face repositories used by the project, the training-only ones and
the unrelated ones are listed in [`HF_REPOS.md`](HF_REPOS.md), so the account
owner can remove everything that is not needed.

For a researcher-supplied model, point `models.model_repo` at an HF **model**
repo that contains the nnU-Net v2 results tree, for example:

```
group00_<DatasetName>/nnUNet_results/Dataset7xx_<DatasetName>/fold_0/...
```

and a groups manifest (`groups_manifest.json` /
`all_groups_manifest.json`). `models.configuration` and `models.fold` must match
the published checkpoint (they are no longer hardcoded to `2d` and `0`).

---

## Use the web form

1. Open the app → the form.
2. Pick one of the three cases, fill in the identifiers.
3. **Preview** validates the spec and shows the exact notebooks and YAML.
4. **Run** executes it; the status card, log and a link to the 3D viewer appear
   as stages finish.

The form writes the same YAML as the CLI. Uploaded archives are sent to
`PFA_UPLOAD_REPO` (defaults to the shared dataset), removed from the web host,
and referenced by URL.

### Where the result is saved

A completed run publishes to the run's `input.hf_repo` (default
`senku111/embryo-3d`):

```
volumes/<specimen-id>/     registered volume + manifest
slices/<specimen-id>/      WebP sections used by the viewer
viewers/<specimen-id>/     manifest.json, meshes.bin, names.json
runs/<run-id>/state.json   run state (PFA_STATE_REPO, default the same repo)
```

The finished status card shows **Open the 3D model**; its URL is
`/viewer/?specimen=<id>&repo=<hf-repo>`. The same card links to the saved bundle
on Hugging Face. The viewer is not limited to one specimen: its picker lists
every `viewers/<id>/manifest.json` in the repo. It shows one entry today because
only `CS17-6521` has a published bundle; a new run adds another entry. A direct
viewer URL works even before the picker refreshes.

The viewer reads the bundle directly from Hugging Face. Use a **public** output
dataset for a shareable viewer link; a private dataset needs authenticated Hub
access and is not readable by the anonymous browser.

### Server environment

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | HF read/write token (also used to read private dataset links) |
| `KAGGLE_API_TOKEN` | Kaggle access token (`KGAT_...`), or `KAGGLE_USERNAME` + `KAGGLE_KEY` |
| `PFA_SHARED_REPO` | shared artifacts repo (default `senku111/embryo-3d`) |
| `PFA_UPLOAD_REPO` | optional separate repo for form uploads |
| `PFA_MAX_UPLOAD_MB` | upload limit (default 2048 MB) |
| `PFA_RUNS_DIR` | local scratch/state directory |
| `PFA_STATE_REPO` | repo for mirrored `runs/<id>/state.json` |

---

## Use the CLI

```bash
pip install papermill nbformat pyyaml huggingface_hub kaggle ipykernel

export HF_TOKEN=hf_...            # read + write the Hugging Face repos
export KAGGLE_API_TOKEN=KGAT_...  # Kaggle access token (username is introspected)
# legacy accounts: export KAGGLE_USERNAME=... KAGGLE_KEY=...

python3 run_pipeline.py --spec specs/cs17_atlas.yaml --dry-run    # preview
python3 run_pipeline.py --spec specs/cs17_atlas.yaml              # run on Kaggle
python3 run_pipeline.py --spec specs/cs17_atlas.yaml --resume     # continue a run
```

Local execution (no Kaggle) uses the notebooks' `WORK_DIR` parameter:

```bash
python3 run_pipeline.py --spec specs/cs17_atlas.yaml --engine local
```

Each run writes `runs/<run_id>/`:

```
spec.yaml        the resolved spec (exact parameters of the run)
state.json       per-stage status, timings, kernel ids, errors
notebooks/       the prepared (parameter-injected) notebooks
logs/<stage>/    the Kaggle kernel output
run.log          the runner's own log
```

---

## Training (report / reproduction only)

Training is not part of the deployment UI. The full training chain is still in
the repository and in [`specs/new_stage_train_from_labels.yaml`](specs/new_stage_train_from_labels.yaml):

```
registration
  → data-preparation            (builds the group datasets from Atlas labels)
  → train group 0..N
  → 3d-need-some-work
```

It is used to reproduce the CS17 project models and to document the method in
the report. A researcher who wants to train a model for case 2/3 runs this chain
separately, publishes the checkpoints as an HF model repo, then comes back to
the deployment form with the model repo.

---

## What is in the box

| Path | Role |
| --- | --- |
| `*.ipynb` | the 7 pipeline notebooks, each with a papermill `parameters` cell |
| `pipeline/params_cells.py` | single source of truth for the parameter cells + defaults |
| `pipeline/spec.py` | spec → validation → ordered list of notebook executions |
| `pipeline/runner.py` | engines (`kaggle`, `local`) + orchestration + artifact checks |
| `pipeline/hfio.py` | Hugging Face helpers (existence checks, uploads, specimen list) |
| `run_pipeline.py` | CLI |
| `tools/parameterize_notebooks.py` | (re)applies the parameter cells; `--check` is CI-friendly |
| `tests/` | parameter-injection and spec/plan tests |
| `space/` | FastAPI app: form → spec → background run, status API, uploads |
| `index5.html` | the 3D viewer (`?specimen=<id>`, `?repo=<dataset>`) |
| `backups/original_notebooks/` | untouched copies of the original notebooks |

The deployment pipeline is

```
registration (part B; validation part A is pruned)
  → 3d-need-some-work           (segmentation + meshing + viewer bundle)
```

HF is the transport between stages: each stage verifies its artifact on the hub
before the next one starts.

---

## Viewer

```
/viewer/?specimen=CS17-6521
/viewer/?specimen=CS12-8505A&repo=my-user/my-specimens
```

The header picker lists every specimen with a `viewers/<id>/manifest.json` in
the shared dataset. Calibration comes from `volumes/<id>/<id>_manifest.json`
(then `slices/<id>/slices_manifest.json`, then `?nz=&ny=&nx=&xy=&z=`).
The bundle title and `model_configuration` / `model_fold` come from
`viewers/<id>/manifest.json`.

Optional per-specimen extras next to `manifest.json`:

* `names.json` — structure names / colours / groups (never overwritten by the pipeline)
* `project.json` — overrides the overview numbers (model groups, held-out dice)
* `morphometrics.json`, `dice_group{0..3}.json` — extra cards

Project Dice/training numbers are only shown for `CS17-6521` or when a
`project.json` is present. A researcher bundle does not inherit the CS17 numbers.

---

## Tests

```bash
python3 tests/test_params.py        # notebooks: cells run together, no stomping, injection
python3 tests/test_pipeline.py      # specs validate, plans are right, defaults inject cleanly
```

The tests do not execute the notebooks; they verify parameter injection, spec
validation and plan ordering.

## Deployment

See [DEPLOY.md](DEPLOY.md) for the full setup. In short:

* **Web app** — Docker web service on **Render Free** from the public GitHub
  repository. Set `HF_TOKEN` and `KAGGLE_API_TOKEN`; the free tier sleeps when
  idle and has an ephemeral disk.
* **After each push** — Render public-URL services need
  **Manual Deploy → Deploy latest commit**; a push alone is not enough.
* **Storage** — Hugging Face Hub (already wired through `HF_TOKEN`).
* **Compute** — the researcher's own Kaggle account (GPU quota), via the API.

Run state is mirrored to `runs/<run_id>/` on the Hub, so a free host that sleeps
resumes unfinished runs instead of losing them.

The Hugging Face Space deployment is optional and currently requires a PRO
account for Docker Spaces.
