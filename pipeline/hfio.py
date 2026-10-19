"""Small Hugging Face helpers used by the runner (existence checks, uploads).

Everything degrades gracefully when no token is available: public reads work
anonymously, and operations that need write access raise a clear error.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_HF = None


def token() -> str | None:
    """Token from the environment (HF_TOKEN) or an existing local login."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import HfFolder  # type: ignore[attr-defined]
        return HfFolder.get_token()
    except Exception:
        pass
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def api():
    global _HF
    if _HF is None:
        from huggingface_hub import HfApi
        _HF = HfApi(token=token())
    return _HF


def reset() -> None:
    """Forget the cached api (after the token changes)."""
    global _HF
    _HF = None


def file_exists(repo_id: str, path: str, repo_type: str = "dataset") -> bool:
    try:
        return bool(api().file_exists(repo_id, path, repo_type=repo_type))
    except Exception:
        return False


def read_json(repo_id: str, path: str, repo_type: str = "dataset") -> Any:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id, path, repo_type=repo_type, token=token())
    return json.loads(Path(p).read_text())


def upload_json(repo_id: str, path: str, obj: Any, repo_type: str = "dataset",
                private: bool = False, commit_message: str = "") -> None:
    """Create the repo if needed and upload a JSON object."""
    a = api()
    a.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    tmp = Path("/tmp") / f"_hf_upload_{Path(path).name}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(obj, indent=2))
    a.upload_file(path_or_fileobj=str(tmp), path_in_repo=path, repo_id=repo_id,
                  repo_type=repo_type, commit_message=commit_message or f"upload {path}")


def list_repo(repo_id: str, repo_type: str = "dataset") -> list[str]:
    try:
        return list(api().list_repo_files(repo_id, repo_type=repo_type))
    except Exception:
        return []


def list_specimens(shared_repo: str = "senku111/embryo-3d") -> list[dict]:
    """Specimens that already have a viewer bundle, for the UI picker."""
    out = []
    files = list_repo(shared_repo, "dataset")
    seen: set[str] = set()
    for f in files:
        parts = f.split("/")
        if len(parts) >= 3 and parts[0] == "viewers" and parts[2] in ("manifest.json", "meshes.bin"):
            seen.add(parts[1])
    for sid in sorted(seen):
        entry = {"id": sid, "viewer": f"https://huggingface.co/datasets/{shared_repo}/resolve/main/viewers/{sid}/"}
        out.append(entry)
    return out


def volume_exists(specimen_id: str, shared_repo: str = "senku111/embryo-3d") -> bool:
    return file_exists(shared_repo, f"volumes/{specimen_id}/{specimen_id}_manifest.json", "dataset")


def groups_manifest_exists(specimen_id: str, dataset_repo: str) -> bool:
    return file_exists(dataset_repo, f"training_dataset/{specimen_id}/reports/all_groups_manifest.json",
                       "dataset")
