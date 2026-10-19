---
title: PFA Embryo 3D Pipeline
emoji: 🥚
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
tags:
  - pipeline
  - nnunet
  -3d
  - segmentation
---

# PFA — 3D embryo reconstruction pipeline

Register a specimen, segment it with nnU-Net group models, and rebuild an
explorable 3D anatomical model: `volumes/<id>/`, `slices/<id>/` and
`viewers/<id>/` on Hugging Face, read by the web viewer.

Everything a researcher has to fill in is a **run spec**. The form on this Space,
the CLI (`run_pipeline.py`) and the notebooks all consume the same spec, so any
run can be replayed exactly.

> `report_pfa.pdf` is the original project report and is being rewritten to
> match this workflow.

---

## Which case are you in?

| Case | Specimen | Segmentation models | Spec |
| --- | --- | --- | --- |
| **1. Reproduce the project** | CS17 atlas specimen `6521` | the pretrained CS17 4-group models | [`specs/cs17_atlas.yaml`](specs/cs17_atlas.yaml) |
| **2. New stage, you already have a model** | your specimen id + Atlas code | your nnU-Net v2 checkpoints in an HF **model** repo | [`specs/new_stage_custom_model.yaml`](specs/new_stage_custom_model.yaml) |
| **3. New stage, nothing trained yet** | your specimen id + Atlas code | trained first from Atlas labels: data-prep → N× training → reconstruction | [`specs/new_stage_train_from_labels.yaml`](specs/new_stage_train_from_labels.yaml) |

Sub-cases:

* **Raw serial sections instead of an Atlas volume** — `input.format: raw_section_archive`
  (+ `xy_pixel_size_um` / `z_spacing_um`, both mandatory).
* **Your own nnU-Net raw dataset instead of Atlas labels** —
  `models.source: train` with `models.dataset_source: nnunet_raw` plus
  `models.groups` and `models.norm`.
* **Private specimen** — `input.hf_private: true` and point `input.hf_repo`
  at a private dataset.

---

## Use the web form

1. Open the Space → the form.
2. Pick one of the three cases, fill in the identifiers.
3. **Plan** shows the exact notebooks and the YAML that will be saved with the run.
4. **Run** (or **Run with resume**) executes it; the status card, log and a link
   to the 3D viewer appear as stages finish.

Everything the form produces is a plain spec — you can paste it into `specs/`
and run it from a terminal instead.

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
| `space/` | FastAPI app: form → spec → background run, status API |
| `index5.html` | the 3D viewer (`?specimen=<id>`, `?repo=<dataset>`) |
| `backups/original_notebooks/` | untouched copies of the original notebooks |

The pipeline order is

```
registration (part B; validation part A is pruned)
  → data-preparation            (only when models.source: train, dataset_source: atlas)
  → train group 0..N            (only when models.source: train)
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

The picker in the header lists every specimen with a `viewers/<id>/manifest.json`
in the shared dataset. Calibration comes from `volumes/<id>/<id>_manifest.json`
(then `slices/<id>/slices_manifest.json`, then `?nz=&ny=&nx=&xy=&z=`).
Optional per-specimen extras next to `manifest.json`:

* `names.json` — structure names / colours / groups (never overwritten by the pipeline)
* `project.json` — overrides the overview numbers (model groups, held-out dice)
* `morphometrics.json`, `dice_group{0..3}.json` — extra cards

---

## Tests

```bash
python3 tests/test_params.py        # notebooks: cells run together, no stomping, injection
python3 tests/test_pipeline.py      # specs validate, plans are right, defaults inject cleanly
```

## Deployment

See [DEPLOY.md](DEPLOY.md) for the recommended deployment (Hugging Face Space +
Kaggle API), the secrets involved and the one-off setup steps.
