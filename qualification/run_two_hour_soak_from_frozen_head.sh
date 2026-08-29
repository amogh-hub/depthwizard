#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HEAD="339bdf485149f552db846543b9e09377b567c19c"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

HEAD="$(git rev-parse HEAD)"
echo "DepthWizard frozen-head soak"
echo "HEAD=$HEAD"

if [[ "$HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "ERROR: checkout is not the qualified production SHA." >&2
  echo "Expected: $EXPECTED_HEAD" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: worktree is not clean. Evidence must be generated from a clean frozen checkout." >&2
  git status --short >&2
  exit 3
fi

APP="apps/desktop/src-tauri/target/release/bundle/macos/DepthWizard.app"
SIDECAR="$APP/Contents/Resources/depthwizard-core-runtime/depthwizard-core"

if [[ ! -d "$APP" ]]; then
  echo "ERROR: packaged DepthWizard.app is missing: $APP" >&2
  exit 4
fi
if [[ ! -x "$SIDECAR" ]]; then
  echo "ERROR: packaged scientific sidecar is missing/not executable: $SIDECAR" >&2
  exit 5
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: .venv/bin/python is missing." >&2
  exit 6
fi

mkdir -p artifacts/acceptance/release-train-7-soak
rm -f artifacts/acceptance/release-train-7-soak/software_stability_report.json

echo "Starting the qualifying 7200-second packaged soak."
.venv/bin/python scripts/release_train_7_soak.py \
  --duration-seconds 7200 \
  --interval-seconds 30 \
  --output-dir artifacts/acceptance/release-train-7-soak

echo "=== SOAK REPORT STATUS ==="
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('artifacts/acceptance/release-train-7-soak/software_stability_report.json')
r = json.loads(p.read_text())
print(json.dumps({
    'status': r.get('status'),
    'git_head': r.get('git_head'),
    'monitored_seconds': r.get('monitored_seconds'),
    'iterations': r.get('iterations'),
    'desktop_memory': r.get('desktop_memory'),
    'sidecar_memory': r.get('sidecar_memory'),
    'request_cycle_seconds': r.get('request_cycle_seconds'),
}, indent=2, sort_keys=True))
assert r.get('status') == 'PASS_TWO_HOUR_PACKAGED_SOAK'
assert r.get('git_head') == '339bdf485149f552db846543b9e09377b567c19c'
assert float(r.get('monitored_seconds', 0)) >= 7200.0
PY

echo "=== UPDATED SIH26175 COMPLETION STATE ==="
.venv/bin/python scripts/check_sih26175_completion.py
