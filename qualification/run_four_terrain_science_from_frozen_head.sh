#!/usr/bin/env bash
set -euo pipefail

FROZEN_HEAD="339bdf485149f552db846543b9e09377b567c19c"
QUAL_REF="origin/qualification/evidence-kit-339bdf4"
REPO="${DEPTHWIZARD_REPO:-/Users/amoghrb/Documents/depthwizard}"
KIT="${TMPDIR:-/tmp}/depthwizard-evidence-kit-339bdf4"

cd "$REPO"

HEAD="$(git rev-parse HEAD)"
if [[ "$HEAD" != "$FROZEN_HEAD" ]]; then
  echo "ERROR: expected frozen production HEAD $FROZEN_HEAD, got $HEAD" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: repository must be clean before final-science evidence generation." >&2
  git status --short >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "ERROR: expected project interpreter at $REPO/.venv/bin/python" >&2
  exit 2
fi
if [[ -z "${NEON_API_TOKEN:-}" ]]; then
  echo "ERROR: NEON_API_TOKEN is required by NEON's authenticated data-download endpoint." >&2
  echo "Export the token in this shell only. Never commit it to GitHub." >&2
  exit 2
fi

# The qualification branch is operations-only. Materialize it outside the repo so the
# exact-head worktree stays clean and all production evidence remains bound to 339bdf4.
git fetch --quiet origin qualification/evidence-kit-339bdf4
rm -rf "$KIT"
mkdir -p "$KIT"
git archive "$QUAL_REF" qualification | tar -x -C "$KIT"

PY="$REPO/.venv/bin/python"
PREFLIGHT_REPORT="$REPO/workspace/final-science-data/input-preflight.json"

run_preflight() {
  "$PY" "$KIT/qualification/preflight_four_terrain_inputs.py" \
    --availability "$REPO/workspace/final-science-data/neon-availability.json" \
    --potsdam-root "$REPO/data/external/isprs-potsdam" \
    --report "$PREFLIGHT_REPORT"
}

printf '\n=== 0/9 METADATA-ONLY INPUT PREFLIGHT ===\n'
if run_preflight; then
  echo "Fresh Potsdam inputs already available locally."
else
  rc=$?
  if [[ "$rc" -ne 4 ]]; then
    echo "ERROR: final-science input preflight failed with unexpected status $rc" >&2
    exit "$rc"
  fi
  printf '\n=== 0A/9 ACQUIRE TWO FRESH OFFICIAL POTSDAM PAIRS ===\n'
  "$PY" "$KIT/qualification/download_potsdam_fresh_pairs_official.py" \
    --tile 2_14 \
    --tile 3_14 \
    --dataset-root "$REPO/data/external/isprs-potsdam" \
    --cache-root "$REPO/workspace/final-science-data/potsdam-official-cache" \
    --report "$REPO/workspace/final-science-data/potsdam-acquisition.json"
  printf '\n=== 0B/9 RE-RUN METADATA-ONLY INPUT PREFLIGHT ===\n'
  run_preflight
fi

printf '\n=== 1/9 ACQUIRE FROZEN CO-ACQUIRED NEON RGB + SEALED DSM REFERENCES ===\n'
"$PY" "$KIT/qualification/download_neon_eval_tiles_locked.py" \
  --output-root "$REPO/workspace/final-science-data/neon" \
  --report "$REPO/workspace/final-science-data/neon-acquisition.json"

printf '\n=== 2/9 FREEZE SIX-SCENE REGISTRY (4 TEST TERRAINS + CROSS-SENSOR + TRAIN LINEAGE) ===\n'
"$PY" "$KIT/qualification/build_final_science_registry.py" \
  --repo "$REPO" \
  --neon-report workspace/final-science-data/neon-acquisition.json \
  --preflight-report workspace/final-science-data/input-preflight.json \
  --potsdam-root data/external/isprs-potsdam \
  --output workspace/final-science-data/frozen-registry.yaml \
  --report workspace/final-science-data/registry-freeze-report.json

printf '\n=== 3/9 GENERATE PRODUCTION DA3 METRIC DSMs WITHOUT OPENING REFERENCES ===\n'
"$PY" "$KIT/qualification/prepare_metric_predictions.py" \
  --repo "$REPO" \
  --registry workspace/final-science-data/frozen-registry.yaml \
  --output-root workspace/final-science-data/predictions \
  --copdem-cache workspace/final-science-data/calibration/copdem-cache \
  --draft workspace/final-science-data/predictions-draft.yaml

printf '\n=== 4/9 FREEZE CHECKPOINT / PREDICTION / CALIBRATION-EVIDENCE IDENTITIES ===\n'
rm -f \
  "$REPO/workspace/final-science-data/frozen-predictions.yaml" \
  "$REPO/workspace/final-science-data/frozen-predictions-freeze-report.json"
"$PY" "$REPO/scripts/freeze_final_science_manifest.py" \
  "$REPO/workspace/final-science-data/frozen-registry.yaml" \
  "$REPO/workspace/final-science-data/predictions-draft.yaml" \
  "$REPO/workspace/final-science-data/frozen-predictions.yaml"

FREEZE_REPORT="$REPO/workspace/final-science-data/frozen-predictions-freeze-report.json"
if [[ ! -f "$FREEZE_REPORT" ]]; then
  echo "ERROR: prediction freeze report was not produced: $FREEZE_REPORT" >&2
  exit 3
fi
"$PY" - "$FREEZE_REPORT" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
assert p["status"] == "PASS_FINAL_SCIENCE_PREDICTION_FREEZE", p
assert p["git_head"] == "339bdf485149f552db846543b9e09377b567c19c", p
assert p["reference_rasters_opened_or_hashed"] is False, p
print("Prediction identity freeze: PASS")
PY

printf '\n=== 5/9 ONLY NOW EXPOSE INDEPENDENT REFERENCES TO FINAL EVALUATOR ===\n'
rm -rf "$REPO/artifacts/final-science"
"$PY" "$REPO/scripts/evaluate_final_science_campaign.py" \
  "$REPO/workspace/final-science-data/frozen-registry.yaml" \
  "$REPO/workspace/final-science-data/frozen-predictions.yaml" \
  "$REPO/artifacts/final-science"

printf '\n=== 6/9 VERIFY FOUR-TERRAIN REPORT CONTRACT ===\n'
REPORT="$REPO/artifacts/final-science/domain_generalization_report.json"
"$PY" - "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
assert p["protocol"] == "depthwizard_final_science_campaign_v1", p.get("protocol")
r = p["requirements"]
assert r["geographic_split_integrity"] == "passed", r
assert r["reference_independence"] == "passed", r
assert set(r["test_terrain_coverage"]) == {"urban", "sparse", "hilly", "forested"}, r
for name in ("urban", "sparse", "hilly", "forested"):
    m = p["terrain"][name]
    assert m["scenes"] >= 1 and m["valid_pixels"] >= 128, (name, m)
    assert all(math.isfinite(float(m[k])) for k in ("rmse_m", "mae_m", "pearson_r")), (name, m)
print("Four-terrain science contract: PASS")
PY

printf '\n=== 7/9 REFRESH SIH26175 COMPLETION STATE ===\n'
"$PY" "$REPO/scripts/check_sih26175_completion.py"

printf '\n============================================================\n'
printf ' FOUR-TERRAIN SCIENCE PIPELINE FINISHED ON FROZEN 339bdf4\n'
printf '============================================================\n'
