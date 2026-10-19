#!/usr/bin/env python3
"""Add the researcher-supplied-model (MODEL_SOURCE='custom') support to the 3D notebook.

Idempotent: refuses to run twice (checks for the new code) unless --force.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "3d-need-some-work.ipynb"

OLD_GROUPS = '''# groups + training normalisation
gm_path = _dl(HF_DATASET_REPO_ID, f"training_dataset/{SPECIMEN_ID}/reports/all_groups_manifest.json", local=BASE / "manifest_dl")
GM = json.load(open(gm_path))
groups = {g["group_index"]: g for g in GM["groups"]}
assert len(groups) == N_GROUPS, f"expected {N_GROUPS} groups, manifest has {len(groups)}"
NORM_LO, NORM_HI = float(GM["normalization"]["global_lo"]), float(GM["normalization"]["global_hi"])
ALL_ATLAS_IDS = sorted(int(a) for g in groups.values() for a in g["atlas_ids"])
print("groups:", {gi: g["n_classes"] for gi, g in groups.items()}, "| training normalisation:", NORM_LO, "-", NORM_HI)'''

NEW_GROUPS = '''# groups + training normalisation
#   MODEL_SOURCE == "existing" -> the manifest comes from the training-dataset repo
#       (training_dataset/<SPECIMEN>/reports/all_groups_manifest.json), as always.
#   MODEL_SOURCE == "custom"   -> the manifest comes from the researcher's own HF model
#       repo (MODEL_REPO), or is passed inline as GROUPS_JSON. Schema is identical to
#       all_groups_manifest.json: {"normalization": {...}, "groups": [{group_index,
#       dataset_id, dataset_name, atlas_ids, n_classes, total_positive_pixels}, ...]}.
#   NORM_OVERRIDE_LO/HI always win when set (must match how the model was trained).
assert MODEL_SOURCE in ("existing", "custom"), f"MODEL_SOURCE must be 'existing' or 'custom', got {MODEL_SOURCE!r}"
if MODEL_SOURCE == "custom":
    assert MODEL_REPO, "MODEL_SOURCE='custom' requires MODEL_REPO (a HF model repo id)"

def _load_groups_manifest():
    if MODEL_SOURCE == "custom":
        if GROUPS_JSON:
            print("groups manifest: inline GROUPS_JSON parameter")
            return json.loads(GROUPS_JSON) if isinstance(GROUPS_JSON, str) else dict(GROUPS_JSON)
        cands = [p for p in [GROUPS_MANIFEST_IN_REPO] if p]
        cands += ["groups_manifest.json", "all_groups_manifest.json",
                  f"training_dataset/{SPECIMEN_ID}/reports/all_groups_manifest.json"]
        for fn in cands:
            try:
                p = _dl(MODEL_REPO, fn, rtype="model", local=BASE / "manifest_dl")
                print(f"groups manifest: {MODEL_REPO} -> {fn}")
                return json.load(open(p))
            except Exception as e:
                print(f"  not available: {MODEL_REPO}/{fn} ({type(e).__name__})")
        raise RuntimeError(
            f"MODEL_SOURCE='custom' found no groups manifest in {MODEL_REPO}. Either upload "
            "groups_manifest.json there (same schema as all_groups_manifest.json) or pass GROUPS_JSON.")
    m_spec = GROUPS_MANIFEST_SPECIMEN_ID or SPECIMEN_ID
    gm_path = _dl(HF_DATASET_REPO_ID, f"training_dataset/{m_spec}/reports/all_groups_manifest.json",
                  local=BASE / "manifest_dl")
    return json.load(open(gm_path))

GM = _load_groups_manifest()
groups = {g["group_index"]: g for g in GM["groups"]}
assert len(groups) == N_GROUPS, f"expected {N_GROUPS} groups, manifest has {len(groups)}"
NORM_LO = float(GM["normalization"]["global_lo"]) if NORM_OVERRIDE_LO is None else float(NORM_OVERRIDE_LO)
NORM_HI = float(GM["normalization"]["global_hi"]) if NORM_OVERRIDE_HI is None else float(NORM_OVERRIDE_HI)
ALL_ATLAS_IDS = sorted(int(a) for g in groups.values() for a in g["atlas_ids"])
print("groups:", {gi: g["n_classes"] for gi, g in groups.items()}, "| training normalisation:", NORM_LO, "-", NORM_HI)'''

OLD_SURVEY_HEAD = '''repo_files = api.list_repo_files(HF_TRAINING_REPO_ID, repo_type="model")
print(len(repo_files), "files in", HF_TRAINING_REPO_ID)

def survey_group(gi):
    g = groups[gi]
    pre = f"group{gi:02d}_{g['dataset_name']}/nnUNet_results/Dataset{g['dataset_id']:03d}_{g['dataset_name']}/"
    runs = {}
    for f in repo_files:
        if not f.startswith(pre): continue
        m = re.match(r"^([^/]+)__([^/]+)__2d/(.*)$", f[len(pre):])
        if m: runs.setdefault((m.group(1), m.group(2)), []).append(m.group(3))
    info = []
    for (tr, pl), files in runs.items():
        fold0 = {x.split("/", 1)[1] for x in files if x.startswith("fold_0/")}
        ep = -1
        for lg in sorted(x for x in files if x.startswith("fold_0/training_log")):
            p = hf_hub_download(HF_TRAINING_REPO_ID, pre + f"{tr}__{pl}__2d/" + lg, repo_type="model", token=HF_TOKEN, local_dir=str(BASE / "logs"))
            for mm in re.finditer(r"Epoch (\\d+)", open(p, errors="ignore").read()): ep = max(ep, int(mm.group(1)) + 1)
        info.append(dict(trainer=tr, plans=pl, final="checkpoint_final.pth" in fold0, best="checkpoint_best.pth" in fold0,
                         latest="checkpoint_latest.pth" in fold0, epochs_logged=ep, prefix=pre))
    return info'''

NEW_SURVEY_HEAD = '''MODEL_REPO_ID = MODEL_REPO if MODEL_SOURCE == "custom" else HF_TRAINING_REPO_ID
repo_files = api.list_repo_files(MODEL_REPO_ID, repo_type="model")
print(len(repo_files), "files in", MODEL_REPO_ID, f"| model source: {MODEL_SOURCE}")

def group_prefix(gi):
    """Folder that holds group gi's runs inside MODEL_REPO_ID.

    Training pushes to `group{gi:02d}_{dataset_name}/nnUNet_results/Dataset{id}_{name}/`.
    A researcher-supplied repo may keep the runs anywhere as long as the
    `Dataset{id:03d}_{name}` folder exists somewhere -- the shortest matching prefix
    wins, so `nnUNet_results/Dataset.../` or a bare `Dataset.../` layout also works.
    """
    g = groups[gi]
    name = f"Dataset{g['dataset_id']:03d}_{g['dataset_name']}/"
    preferred = f"group{gi:02d}_{g['dataset_name']}/nnUNet_results/{name}"
    if any(f.startswith(preferred) for f in repo_files):
        return preferred
    hits = [f[: f.find(name) + len(name)] for f in repo_files if f.find(name) >= 0]
    if hits:
        pre = min(hits, key=len)
        print(f"  group {gi}: repo layout -> {pre}")
        return pre
    return preferred

def survey_group(gi):
    g = groups[gi]
    pre = group_prefix(gi)
    runs = {}
    for f in repo_files:
        if not f.startswith(pre): continue
        rest = f[len(pre):]
        m = re.search(r"([^/]+)__([^/]+)__2d/(.*)$", rest)
        if not m: continue
        run_dir = pre + rest[:m.start()] + f"{m.group(1)}__{m.group(2)}__2d/"
        runs.setdefault((m.group(1), m.group(2)), {"files": [], "run_dir": run_dir})["files"].append(m.group(3))
    info = []
    for (tr, pl), run in runs.items():
        files = run["files"]
        fold0 = {x.split("/", 1)[1] for x in files if x.startswith("fold_0/")}
        ep = -1
        for lg in sorted(x for x in files if x.startswith("fold_0/training_log")):
            p = hf_hub_download(MODEL_REPO_ID, run["run_dir"] + lg, repo_type="model", token=HF_TOKEN, local_dir=str(BASE / "logs"))
            for mm in re.finditer(r"Epoch (\\d+)", open(p, errors="ignore").read()): ep = max(ep, int(mm.group(1)) + 1)
        info.append(dict(trainer=tr, plans=pl, final="checkpoint_final.pth" in fold0, best="checkpoint_best.pth" in fold0,
                         latest="checkpoint_latest.pth" in fold0, epochs_logged=ep, prefix=pre, run_dir=run["run_dir"]))
    return info'''

OLD_CHOOSE = '''    pol = CHECKPOINT_OVERRIDE.get(gi, CHECKPOINT_POLICY)'''
NEW_CHOOSE = '''    pol = CHECKPOINT_OVERRIDE.get(gi, CHECKPOINT_OVERRIDE.get(str(gi), CHECKPOINT_POLICY))'''

OLD_DL = '''    for rel in ["dataset.json", "plans.json", f"fold_0/checkpoint_{ck}.pth"]:
        src = run["prefix"] + f"{run['trainer']}__{run['plans']}__2d/" + rel
        if src not in repo_files:
            src = run["prefix"] + rel
            assert src in repo_files, f"group {gi}: missing {rel} in repo"
        p = hf_hub_download(HF_TRAINING_REPO_ID, src, repo_type="model", token=HF_TOKEN, local_dir=str(BASE / "dl"))
        dst = mdir / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)'''
NEW_DL = '''    for rel in ["dataset.json", "plans.json", f"fold_0/checkpoint_{ck}.pth"]:
        src = run["run_dir"] + rel
        if src not in repo_files:
            src = run["prefix"] + f"{run['trainer']}__{run['plans']}__2d/" + rel
        if src not in repo_files:
            src = run["prefix"] + rel
        assert src in repo_files, f"group {gi}: missing {rel} in {MODEL_REPO_ID}"
        p = hf_hub_download(MODEL_REPO_ID, src, repo_type="model", token=HF_TOKEN, local_dir=str(BASE / "dl"))
        dst = mdir / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)'''


def main() -> None:
    force = "--force" in sys.argv
    nb = nbf.read(str(NB), as_version=4)
    if not force and any(
        "def group_prefix(" in "".join(c.source) for c in nb.cells if c.cell_type == "code"
    ):
        print("already applied")
        return
    edits = 0

    def cell_index(needle: str) -> int:
        for i, c in enumerate(nb.cells):
            if c.cell_type == "code" and needle in "".join(c.source):
                return i
        raise SystemExit(f"cell containing {needle!r} not found")

    targets = [
        (cell_index("# groups + training normalisation"), OLD_GROUPS, NEW_GROUPS),
        (cell_index("repo_files = api.list_repo_files"), OLD_SURVEY_HEAD, NEW_SURVEY_HEAD),
        (cell_index("repo_files = api.list_repo_files"), OLD_CHOOSE, NEW_CHOOSE),
        (cell_index("repo_files = api.list_repo_files"), OLD_DL, NEW_DL),
    ]
    for cell, old, new in targets:
        src = "".join(nb.cells[cell].source)
        if old not in src:
            if new in src:
                continue
            sys.exit(f"anchor not found in cell {cell}:\n{old[:160]}")
        nb.cells[cell].source = src.replace(old, new, 1)
        edits += 1
    nbf.write(nb, str(NB))
    print(f"applied {edits} edits")


if __name__ == "__main__":
    main()
