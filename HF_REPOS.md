# HistoWeave — Hugging Face repository inventory

Checked against the `senku111` account on 2026-10-26. Nothing was deleted.
The account currently exposes **5 models, 14 datasets and 0 Spaces** publicly;
a private dataset used by the project does not appear in an anonymous listing.

## Used by the deployment — keep

| Repository | Type | Why it is used |
| --- | --- | --- |
| `senku111/embryo-3d` | dataset (public) | Shared artifact store: `volumes/<id>/`, `slices/<id>/`, `viewers/<id>/`, `runs/<id>/state.json`, and form uploads. The viewer and the specimen picker read it directly. |
| `senku111/atlas-cs17-4groups-nnunet-training` | model (public) | The four CS17 nnU-Net checkpoints and their `dataset.json` files. This is the model used by **case 1** (same specimen, project model). |
| `senku111/atlas-cs17-4groups-dataset` | dataset (private / not anonymously visible) | Canonical training dataset and `all_groups_manifest.json` for the CS17 four-group models. **Case 1 no longer requires it**: the reconstruction notebook can derive the group layout from the `dataset.json` files in the checkpoint repo. Keep it if you want to reproduce or retrain the project models. |

## Training and report only — not used by the deployed app

These are checkpoints or datasets from the experiments described in the report.
The web form never reads them; cases 2 and 3 use the researcher's own model repo.
They can be deleted if you do not need to rerun the training or regenerate the
report figures.

| Repository | Type | Role |
| --- | --- | --- |
| `senku111/atlas-cs15-nnunet-training` | model | Early CS15 registration/segmentation experiment. |
| `senku111/atlas-cs15-nnunet-training-v2` | model | CS15 experiment, second iteration. |
| `senku111/atlas-cs15-nnunet-training-v3` | model | CS15 experiment, third iteration. |
| `senku111/atlas-cs15-nnunet` | dataset | CS15 training data. |
| `senku111/atlas-cs17-nnunet-training-v1` | model | Early CS17 single-model attempt (not in the current public listing; may be private or already removed). |
| `senku111/atlas-cs17-nnunet-training-v3-new` | model | CS17 experiment after the group redesign. |
| `senku111/atlas-cs17-nnunet-single-model-training-v1` | model | Single-model ablation (not in the current public listing). |
| `senku111/atlas-cs17-nnunet-single-model-data-v1` | dataset | Data for the single-model ablation. |

## Obsolete — safe to delete after checking the GitHub deployment

| Repository | Type | Why it is obsolete |
| --- | --- | --- |
| `senku111/pfa-source` | dataset (public) | Old Hugging Face source mirror. Render cannot build from a dataset URL; the current deployment uses `github.com/EngSenku/pfa-embryo-3d`. Nothing in the app reads this repo. |

## Unrelated to this project — safe to delete if no other project needs them

These appeared in the public `senku111` listing but are not referenced by the
HistoWeave code, notebooks, report, `render.yaml`, or the deployed app.

| Repository | Type |
| --- | --- |
| `senku111/3d-module` | dataset |
| `senku111/3d-module-new` | dataset |
| `senku111/books-storage` | dataset |
| `senku111/manga` | dataset |
| `senku111/colon-3d-crc1` | dataset |
| `senku111/crc1-3d-histology-module` | dataset |
| `senku111/crc1-3d-histology-module-v2` | dataset |
| `senku111/crc1-3d-histology-site` | dataset |
| `senku111/crc1-valis-results` | dataset |
| `senku111/crc1-zeroreg-results` | dataset |

`qrayti_*` repositories were mentioned in an earlier inventory but are not in
the current public listing; if they still exist, they are unrelated too.

## Notes

* `senku111/histoweave` is the optional Hugging Face Space name used by
  `tools/deploy_space.sh`; no Space exists on the account yet.
* There are no Spaces on the account.
* Deleting `atlas-cs17-4groups-nnunet-training` or `embryo-3d` breaks the
  deployed app. Deleting `pfa-source` or the training-only repos does not.
* The upload endpoint writes to `PFA_UPLOAD_REPO`, which defaults to
  `senku111/embryo-3d`. If you delete that repo, also change the Render env vars
  `PFA_SHARED_REPO`, `PFA_UPLOAD_REPO` and the form's default dataset.
