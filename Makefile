.PHONY: test verify service frontend-build rust-verify dependency-locks reproducibility-audit supply-chain-report da3-setup da3-smoke height-model-smoke height-train-acceptance height-multiscene-v2 height-multiscene-v3 height-multiscene-v4 height-multiscene-acceptance height-adaptive-acceptance height-frozen-holdout-v1 height-frozen-holdout potsdam-contract-audit potsdam-external-v1 potsdam-external-v2-preflight potsdam-external-v2-execution-preflight potsdam-external-acceptance release-train-2-validation-smoke release-train-3-spatial-foundation-smoke release-train-3-mesh-smoke release-train-3-export-smoke release-train-3-workstation-smoke release-train-3-acceptance release-train-4-analytical-smoke sidecar-build release-train-5-sidecar-smoke release-train-5-app-smoke release-train-5-full-smoke release-train-5-standalone-acceptance release-train-5-workstation-build release-train-5-final-qualification standalone-build demo-rdsm demo-mesh demo-ui demo-india-absolute benchmark-ortholoc-demo rdah-setup benchmark-rdah-ortholoc rdah-sweep-setup benchmark-rdah-sweep

test:
	python -m pytest

verify:
	python scripts/verify.py

service:
	depthwizard serve --host 127.0.0.1 --port 8765

frontend-build:
	cd apps/desktop && npm ci --no-audit --no-fund && npm run build

rust-verify:
	python -m scripts.ensure_tauri_sidecar_stub
	cd apps/desktop/src-tauri && cargo fmt --check && cargo clippy --locked --all-targets --all-features -- -D warnings && cargo test --locked --all-targets --all-features

# Generate resolver outputs from the exact checked-out manifests. These files are release inputs and
# must be reviewed/committed before the strict reproducibility audit can pass. This target resolves
# dependencies; it is intentionally separate from ordinary verification so source tests never
# silently mutate release dependency state.
dependency-locks:
	uv lock
	cd apps/desktop && npm install --package-lock-only --ignore-scripts --no-audit --no-fund
	cd apps/desktop/src-tauri && cargo generate-lockfile

reproducibility-audit:
	python -m scripts.check_release_reproducibility --strict

supply-chain-report:
	python -m scripts.generate_supply_chain_reports

da3-setup:
	bash scripts/setup_da3_macos.sh

da3-smoke:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/smoke_da3.py

height-model-smoke:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/smoke_height_model.py

height-train-acceptance:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train_ortholoc_height_acceptance.py

height-multiscene-v2:
	DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.train_ortholoc_multiscene_v2

height-multiscene-v3:
	DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.train_ortholoc_multiscene_v3

height-multiscene-v4:
	DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.train_ortholoc_multiscene_v4

height-multiscene-acceptance: height-multiscene-v4

height-adaptive-acceptance:
	DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.evaluate_ortholoc_adaptive_refinement

height-frozen-holdout-v1:
	DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.evaluate_ortholoc_frozen_holdout

height-frozen-holdout:
	DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.evaluate_ortholoc_frozen_location_v2

potsdam-contract-audit:
	python -m scripts.audit_potsdam_contract

potsdam-external-v1:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.evaluate_potsdam_external

potsdam-external-v2-preflight:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.evaluate_potsdam_external_v2 --preflight-only

potsdam-external-v2-execution-preflight:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.preflight_potsdam_external_v2_execution

potsdam-external-acceptance:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.evaluate_potsdam_external_v2

release-train-2-validation-smoke:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.release_train_2_validation_smoke

release-train-3-spatial-foundation-smoke:
	DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.release_train_3_spatial_foundation_smoke

release-train-3-mesh-smoke:
	DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE=1 python -m scripts.release_train_3_mesh_smoke

release-train-3-export-smoke:
	python -m scripts.release_train_3_export_smoke

release-train-3-workstation-smoke:
	DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE=1 python -m scripts.release_train_3_workstation_smoke

release-train-3-acceptance: release-train-3-spatial-foundation-smoke release-train-3-mesh-smoke release-train-3-export-smoke release-train-3-workstation-smoke

release-train-4-analytical-smoke:
	python -m scripts.release_train_4_scientific_analytical_smoke

# The production sidecar imports the frozen monocular model/height-model stack, so a clean build
# must install the ML extra explicitly rather than relying on torch already being present locally.
sidecar-build:
	uv sync --frozen --python 3.12 --extra ml --extra standalone
	.venv/bin/python -m scripts.build_standalone_sidecar

release-train-5-sidecar-smoke:
	python -m scripts.release_train_5_sidecar_smoke

release-train-5-app-smoke:
	python -m scripts.release_train_5_app_bundle_smoke

release-train-5-full-smoke:
	python -m scripts.release_train_5_full_acceptance

release-train-5-standalone-acceptance: release-train-5-sidecar-smoke release-train-5-app-smoke release-train-5-full-smoke

# Workstation correction packaging must rebuild the Python sidecar from the exact checked-out
# source before Tauri bundles it. This prevents a new React desktop from silently shipping against
# an older FastAPI API surface (for example missing preview/legend workstation routes).
# It deliberately avoids rerunning the already-earned heavy RT5 scientific acceptance campaign.
release-train-5-workstation-build:
	uv sync --frozen --python 3.12 --extra ml --extra dev --extra standalone
	$(MAKE) verify
	.venv/bin/python -m scripts.build_standalone_sidecar
	cd apps/desktop && npm ci --no-audit --no-fund && npm test && npm run tauri build

# Final qualification is fail-closed on all three committed resolver locks. `uv lock --check`
# proves pyproject.toml and uv.lock still describe the same Python resolution before `uv sync
# --frozen` creates/updates the project .venv strictly from uv.lock. The ML extra contains the full
# curated DA3 monocular runtime closure; the explicit dependency import preflight prevents an
# incomplete scientific environment from reaching PyInstaller. npm uses `ci`; Rust verification
# uses Cargo's `--locked` mode. Tauri then builds against that already-verified graph, and the
# post-build diff check proves none of the committed resolver inputs drifted during packaging.
release-train-5-final-qualification:
	test -f uv.lock
	test -f apps/desktop/package-lock.json
	test -f apps/desktop/src-tauri/Cargo.lock
	git ls-files --error-unmatch uv.lock apps/desktop/package-lock.json apps/desktop/src-tauri/Cargo.lock >/dev/null
	test -z "$$(git status --porcelain)"
	uv lock --check
	uv sync --frozen --python 3.12 --extra ml --extra dev --extra standalone
	.venv/bin/python -c "from depthwizard.geometry_prior.da3_runtime_contract import verify_da3_runtime_dependencies; modules = verify_da3_runtime_dependencies(); print('DA3 runtime dependency imports PASS:', len(modules))"
	.venv/bin/python -c "import torch, torchvision; from torch import nn; assert nn.Module is not None; print('Scientific runtime imports PASS:', torch.__version__, torchvision.__version__)"
	.venv/bin/python -m scripts.check_release_reproducibility --strict
	.venv/bin/python scripts/verify.py
	.venv/bin/python -m scripts.ensure_tauri_sidecar_stub
	cd apps/desktop && npm ci --no-audit --no-fund && npm test && npm run build
	cd apps/desktop/src-tauri && cargo fmt --check && cargo clippy --locked --all-targets --all-features -- -D warnings && cargo test --locked --all-targets --all-features
	.venv/bin/python -m scripts.build_standalone_sidecar
	cd apps/desktop && npm run tauri build
	git diff --exit-code -- uv.lock apps/desktop/package-lock.json apps/desktop/src-tauri/Cargo.lock
	.venv/bin/python -m scripts.release_train_5_sidecar_smoke
	.venv/bin/python -m scripts.release_train_5_app_bundle_smoke
	.venv/bin/python -m scripts.release_train_5_full_acceptance

standalone-build: sidecar-build frontend-build
	cd apps/desktop && npm run tauri build

demo-rdsm:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/demo_geotiff.py

demo-mesh:
	depthwizard mesh-rdsm artifacts/demo/geotiff-rdsm/rdsm.tif data/demo/RGB.byte.tif artifacts/demo/geotiff-mesh --vertical-scale 5000

demo-ui:
	bash scripts/demo_ui.sh

demo-india-absolute:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/demo_india_absolute.py

benchmark-ortholoc-demo:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/benchmark_ortholoc_demo.py

rdah-setup:
	python scripts/setup_rdah_baseline.py

benchmark-rdah-ortholoc:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/benchmark_rdah_ortholoc.py

rdah-sweep-setup:
	python scripts/setup_rdah_checkpoint_sweep.py

benchmark-rdah-sweep:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/benchmark_rdah_checkpoint_sweep.py
