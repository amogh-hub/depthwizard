#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HEAD="2ec539aea010b974a2781b240fe76c67a99d06a8"

if [[ $# -ne 3 ]]; then
  cat >&2 <<'EOF'
Usage:
  stage_manual_evidence_from_frozen_head.sh \
    /path/to/operator-acceptance.json \
    /path/to/rendering-performance.json \
    /path/to/clean-machine-standalone.json

The three files must contain real observed evidence. This helper does not turn pending templates into PASS.
EOF
  exit 64
fi

OPERATOR="$1"
PERFORMANCE="$2"
CLEAN_MACHINE="$3"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "ERROR: checkout is not the qualified corrected source SHA." >&2
  echo "Expected: $EXPECTED_HEAD" >&2
  echo "Actual:   $HEAD" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: worktree is not clean." >&2
  git status --short >&2
  exit 3
fi

for f in "$OPERATOR" "$PERFORMANCE" "$CLEAN_MACHINE"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: evidence file does not exist: $f" >&2
    exit 4
  fi
done

.venv/bin/python - "$OPERATOR" "$PERFORMANCE" "$CLEAN_MACHINE" <<'PY'
import json
import sys
from pathlib import Path
expected = '2ec539aea010b974a2781b240fe76c67a99d06a8'
expected_status = {
    'operator': 'PASS_SIH26175_OPERATOR_ACCEPTANCE',
    'performance': 'PASS_SUSTAINED_3D_PERFORMANCE',
    'clean': 'PASS_CLEAN_MACHINE_STANDALONE',
}
for label, raw in zip(('operator','performance','clean'), sys.argv[1:]):
    p = Path(raw)
    r = json.loads(p.read_text())
    if r.get('git_head') != expected:
        raise SystemExit(f'{label}: wrong git_head: {r.get("git_head")!r}')
    if r.get('status') != expected_status[label]:
        raise SystemExit(
            f'{label}: status is not the required PASS literal: {r.get("status")!r}'
        )
    print(f'{label}: top-level identity/status precheck PASS')
PY

DEST="artifacts/acceptance/sih26175"
mkdir -p "$DEST"
cp "$OPERATOR" "$DEST/operator-acceptance.json"
cp "$PERFORMANCE" "$DEST/rendering-performance.json"
cp "$CLEAN_MACHINE" "$DEST/clean-machine-standalone.json"

# The authoritative checker performs the full field-level validation.
.venv/bin/python scripts/check_sih26175_completion.py
