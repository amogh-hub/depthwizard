#!/usr/bin/env bash
set -euo pipefail

PRODUCTION_HEAD="339bdf485149f552db846543b9e09377b567c19c"
HELPER_REF="qualification/final-evidence-orchestration:qualification/run_final_science_stage1.py"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$PRODUCTION_HEAD" ]]; then
  echo "REFUSED: worktree must remain on frozen production head $PRODUCTION_HEAD; got $HEAD" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "REFUSED: frozen production worktree is not clean" >&2
  exit 1
fi

# Fetch only the qualification branch pointer. This does not move or modify the production branch.
git fetch origin qualification/final-evidence-orchestration

mkdir -p artifacts
HELPER="$ROOT/artifacts/run_final_science_stage1.py"
git show "origin/$HELPER_REF" > "$HELPER"

# artifacts/ is ignored, so materializing the helper here preserves exact-head cleanliness.
test -z "$(git status --porcelain)"

exec "$ROOT/.venv/bin/python" "$HELPER"
