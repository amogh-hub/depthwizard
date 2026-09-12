# Authoritative release playbook

DepthWizard has one guarded release path. Engineering branches may contain stronger work than
`main`, but no submission is final until an exact candidate is independently qualified, identified
by a tag and published with checksum-verifiable artifacts. The completed SIH release is
[`v0.2.0-sih-final`](https://github.com/amogh-hub/depthwizard/releases/tag/v0.2.0-sih-final), sourced
from `012301b9c1910ef4ccde5b3da5d4e4d94da60ce6`.

## 1. Protect `main`

Enable branch protection for `main`, require pull requests, dismiss stale approvals, require the
following current checks, require branches to be up to date and block force pushes/deletion:

- dependency lock integrity
- python-core (3.12)
- python-core (3.13)
- desktop-frontend
- desktop-rust
- portable geospatial (macos-15)
- portable geospatial (windows-2025)
- SBOM and dependency license inventory

Repository administrators must verify these settings in GitHub; source code cannot enforce hosting
policy by itself.

## 2. Freeze and qualify one candidate

Use `engineering/elite-finalization`. Stop source changes, record the exact 40-character SHA, and
rerun source, packaged RT5, literal input, final science, baseline/ablation, operator, performance,
two-hour soak and clean-machine qualification on that SHA. Commit only the compact reports listed
in `evidence/submission/README.md` to the separate branch
`qualification/evidence-<40-character-candidate-SHA>`; never add the reports to the candidate and
never commit private/large raw datasets. Keeping evidence on a separate ref avoids the impossible
self-reference that would result if evidence claimed the SHA of a commit that also contained it.

Run the aggregate gate in a clean checkout of the candidate, pointing each argument at the evidence
branch checkout:

```bash
EVIDENCE=/path/to/qualification-checkout/evidence/submission
python -m scripts.check_sih26175_completion \
  --strict \
  --rt5 "$EVIDENCE/release-train-5-full-acceptance.json" \
  --inputs "$EVIDENCE/input-format-contract.json" \
  --science "$EVIDENCE/domain_generalization_report.json" \
  --baselines "$EVIDENCE/baseline-comparison.json" \
  --ablations "$EVIDENCE/ablation-report.json" \
  --soak "$EVIDENCE/software_stability_report.json" \
  --operator "$EVIDENCE/operator-acceptance.json" \
  --performance "$EVIDENCE/rendering-performance.json" \
  --clean-machine "$EVIDENCE/clean-machine-standalone.json"
```

## 3. Merge, tag and publish

Merge the qualified PR without changing its tree. Confirm `main` equals the qualified SHA, then tag
that exact commit and push the tag. The matching qualification
branch must already exist. `.github/workflows/release.yml` fetches and records that evidence commit,
repeats mandatory verification, rebuilds the standalone application, produces the macOS installer,
checksums, model manifest, completion report, CycloneDX SBOM and dependency-license inventory, then
publishes a checksum-verifiable GitHub Release. Repository administrators must separately enable
GitHub's immutable-release setting if their plan and hosting policy support it.

If a future workflow fails, do not publish an unqualified rebuild around it. Fix the defect on a new
candidate commit, invalidate affected evidence and rerun the required qualification.

For `v0.2.0-sih-final`, the first tag run failed before publication because its Rust preflight ran
before the Tauri compile-only runtime resource was staged. A separate fail-closed workflow verified
the exact source, tag, evidence and prequalified DMG hash, required all eleven completion gates,
published that exact DMG, then re-downloaded and checksum-verified every asset. The workflow ordering
is corrected for future tags; the incident and recovery are recorded in
`docs/elite-finalization-status.md`.

## 4. Judge rehearsal

On a fresh supported Mac, download only the published Release, verify `SHA256SUMS`, disconnect the
network, install and record the complete PNG/JPG/GeoTIFF, DEM/GCP, navigation, measurement, export,
relaunch and repeated-job workflow. The recording, screenshots and presentation must name the tag,
Git SHA, model hash and hardware.
