#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HEAD="339bdf485149f552db846543b9e09377b567c19c"

if [[ $# -ne 4 ]]; then
  cat >&2 <<'EOF'
Usage:
  run_final_science_from_frozen_head.sh \
    /path/to/frozen-registry.yaml \
    /path/to/draft-predictions.yaml \
    /path/to/frozen-predictions.yaml \
    /path/to/output-dir
EOF
  exit 64
fi

REGISTRY="$1"
DRAFT="$2"
FROZEN="$3"
OUT="$4"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

HEAD="$(git rev-parse HEAD)"
echo "DepthWizard final-science campaign"
echo "HEAD=$HEAD"

if [[ "$HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "ERROR: checkout is not the qualified production SHA." >&2
  echo "Expected: $EXPECTED_HEAD" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: worktree is not clean. Freeze/evaluation must run from the clean frozen checkout." >&2
  git status --short >&2
  exit 3
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: .venv/bin/python is missing." >&2
  exit 4
fi

for f in "$REGISTRY" "$DRAFT"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required file is missing: $f" >&2
    exit 5
  fi
done

mkdir -p "$(dirname "$FROZEN")" "$OUT"

# Step 1: freeze checkpoint, predictions and calibration-evidence byte identities.
# The frozen implementation explicitly does not open/hash evaluation reference rasters here.
.venv/bin/python scripts/freeze_final_science_manifest.py \
  "$REGISTRY" \
  "$DRAFT" \
  "$FROZEN"

FREEZE_REPORT="$(dirname "$FROZEN")/$(basename "${FROZEN%.*}")-freeze-report.json"
if [[ ! -f "$FREEZE_REPORT" ]]; then
  echo "ERROR: expected freeze report is missing: $FREEZE_REPORT" >&2
  exit 6
fi

.venv/bin/python - "$FREEZE_REPORT" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
r = json.loads(p.read_text())
assert r.get('status') == 'PASS_FINAL_SCIENCE_PREDICTION_FREEZE', r
assert r.get('git_head') == '339bdf485149f552db846543b9e09377b567c19c', r
assert r.get('reference_rasters_opened_or_hashed') is False, r
print('Prediction identity freeze: PASS')
print('Frozen manifest SHA-256:', r.get('frozen_manifest_sha256'))
print('Evaluation scene count:', r.get('evaluation_scene_count'))
PY

# Step 2: only after the freeze succeeds, expose references to the evaluator.
.venv/bin/python scripts/evaluate_final_science_campaign.py \
  "$REGISTRY" \
  "$FROZEN" \
  "$OUT"

REPORT="$OUT/domain_generalization_report.json"
if [[ ! -f "$REPORT" ]]; then
  echo "ERROR: final science report was not produced: $REPORT" >&2
  exit 7
fi

.venv/bin/python - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
r = json.loads(p.read_text())
assert r.get('protocol') == 'depthwizard_final_science_campaign_v1', r
req = r.get('requirements') or {}
assert req.get('geographic_split_integrity') == 'passed', req
assert req.get('reference_independence') == 'passed', req
assert set(req.get('test_terrain_coverage') or []) == {'urban','sparse','hilly','forested'}, req
print('Final science protocol:', r.get('protocol'))
print('Four-terrain coverage: PASS')
print('Geographic split integrity: PASS')
print('Reference independence: PASS')
print(json.dumps({
    'test_overall': r.get('test_overall'),
    'cross_sensor_overall': r.get('cross_sensor_overall'),
    'terrain': r.get('terrain'),
}, indent=2, sort_keys=True))
PY

echo "=== UPDATED SIH26175 COMPLETION STATE ==="
.venv/bin/python scripts/check_sih26175_completion.py
