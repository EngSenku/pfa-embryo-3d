#!/usr/bin/env bash
# Publish a clean, anonymous-clonable mirror of this repo to the Hugging Face
# Hub.  Render can build a free web service straight from the mirror URL, so a
# GitHub account is not required:
#
#   https://huggingface.co/datasets/senku111/pfa-source
#
# The mirror is a single fresh commit containing the tracked files.  The stale
# report PDF is skipped because the Hub now stores binaries through Xet/LFS and
# rejects a plain git push that contains one; the report is not needed to build
# or run the pipeline.
#
# Usage:
#   export HF_TOKEN=hf_...
#   tools/publish_source.sh [owner/repo]        # default senku111/pfa-source
#
set -euo pipefail

REPO="${1:-senku111/pfa-source}"
: "${HF_TOKEN:?export HF_TOKEN=hf_... (write access to $REPO)}"

ROOT="$(git rev-parse --show-toplevel)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$ROOT" archive HEAD | tar -x -C "$TMP"
rm -f "$TMP/report_pfa.pdf"

# A sane identity is required in the throwaway repo.
export GIT_AUTHOR_NAME="${GIT_AUTHOR_NAME:-PFA}"
export GIT_AUTHOR_EMAIL="${GIT_AUTHOR_EMAIL:-pfa@localhost}"
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

git -C "$TMP" init -q -b main
git -C "$TMP" add -A
git -C "$TMP" commit -q -m "PFA source mirror @ ${SHA}"

echo "publishing ${SHA} to https://huggingface.co/datasets/${REPO} …"
git -C "$TMP" push --force \
  "https://senku111:${HF_TOKEN}@huggingface.co/datasets/${REPO}" main:main

echo "done.  In Render: Manual Deploy → Deploy latest commit."
