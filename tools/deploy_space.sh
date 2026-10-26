#!/usr/bin/env bash
# Create the Hugging Face Space and push the committed project to it.
#
#   export HF_TOKEN=hf_...
#   tools/deploy_space.sh [space-id]        # default: senku111/histoweave
#
# Docker Spaces require a PRO plan on the account (Static Spaces are the only
# free ones); the create call returns HTTP 402 until then.
set -euo pipefail

SPACE_ID="${1:-senku111/histoweave}"
: "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
cd "$(dirname "$0")/.."

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "this directory is not a git repo — run 'git init && git add -A && git commit' first" >&2
  exit 1
fi

python3 - "$SPACE_ID" <<'PY'
import os, sys
from huggingface_hub import HfApi
sid = sys.argv[1]
print("creating:", sid)
url = HfApi(token=os.environ["HF_TOKEN"]).create_repo(
    sid, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
print("space ready:", url)
PY

# Push without persisting the token in .git/config.
git push "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_ID}" HEAD:main

echo
echo "pushed. Build the Space and watch it at:"
echo "  https://huggingface.co/spaces/${SPACE_ID}"
echo "Add secrets there: HF_TOKEN and KAGGLE_API_TOKEN (Settings -> Variables and secrets)."
