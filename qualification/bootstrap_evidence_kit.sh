#!/usr/bin/env bash
set -euo pipefail

FROZEN_HEAD="339bdf485149f552db846543b9e09377b567c19c"
REPO="${DEPTHWIZARD_REPO:-/Users/amoghrb/Documents/depthwizard}"
QUAL_BRANCH="qualification/evidence-kit-339bdf4"
KIT="${TMPDIR:-/tmp}/depthwizard-evidence-kit-339bdf4"

cd "$REPO"
if [[ "$(git rev-parse HEAD)" != "$FROZEN_HEAD" ]]; then
  echo "ERROR: checkout is not the frozen DepthWizard head $FROZEN_HEAD" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: checkout must be clean before qualification." >&2
  git status --short >&2
  exit 2
fi
if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  echo "ERROR: expected project Python at $REPO/.venv/bin/python" >&2
  exit 2
fi

git fetch --quiet origin "$QUAL_BRANCH"
rm -rf "$KIT"
mkdir -p "$KIT"
git archive "origin/$QUAL_BRANCH" qualification | tar -x -C "$KIT"

# Catch Python, shell and template syntax errors before any network/model/long-duration work.
"$REPO/.venv/bin/python" "$KIT/qualification/selfcheck_evidence_kit.py" \
  --kit-root "$KIT/qualification"

cat <<EOF
DepthWizard evidence kit materialized and syntax-checked without changing the production checkout.

Frozen production HEAD:
  $FROZEN_HEAD

Temporary evidence-kit path:
  $KIT/qualification

Useful commands:

  # Resolve public NEON common acquisitions (metadata only)
  $REPO/.venv/bin/python $KIT/qualification/neon_common_acquisitions.py

  # Four-terrain science (requires NEON_API_TOKEN and unused Potsdam 2_14 + 3_14 files)
  bash $KIT/qualification/run_four_terrain_science_from_frozen_head.sh

  # Full qualifying two-hour packaged soak
  bash $KIT/qualification/run_two_hour_soak_from_frozen_head.sh

  # Stage real operator/FPS/clean-machine evidence after observation
  bash $KIT/qualification/stage_manual_evidence_from_frozen_head.sh --help

The qualification branch is operations-only. Do not checkout or merge it into the frozen
production branch while the current RT5 evidence is authoritative.
EOF
