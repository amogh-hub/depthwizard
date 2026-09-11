# Exact-commit submission evidence

This directory defines the fail-closed evidence-pack contract. Final JSON files do **not** live in
the release candidate's own commit: doing that would change the commit SHA and make an exact-head
claim self-contradictory. After a candidate is frozen, produce the reports from that exact SHA and
commit them on the separate branch `qualification/evidence-<40-character-candidate-SHA>`, retaining
this `evidence/submission` layout. Never copy an older report forward by editing its `git_head`, and
never reconstruct a missing measurement from notes or conversational memory.

The tag-triggered release workflow requires:

- `release-train-5-full-acceptance.json`
- `input-format-contract.json`
- `domain_generalization_report.json`
- `baseline-comparison.json`
- `ablation-report.json`
- `software_stability_report.json`
- `operator-acceptance.json`
- `rendering-performance.json`
- `clean-machine-standalone.json`

The tag-triggered release workflow derives the qualification branch name from the tagged SHA,
archives this directory outside the source worktree, records the qualification commit, and runs
`scripts/check_sih26175_completion.py --strict`. The checker validates status, required measurements
and exact source identity before an installer can be published. Large source datasets and model
weights do not belong here; reports must contain their immutable hashes and provenance.
