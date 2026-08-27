.PHONY: test verify service frontend-build rust-verify da3-setup da3-smoke height-model-smoke height-train-acceptance height-multiscene-v2 height-multiscene-v3 height-multiscene-v4 height-multiscene-acceptance height-adaptive-acceptance height-frozen-holdout-v1 height-frozen-holdout potsdam-contract-audit potsdam-external-v1 potsdam-external-v2-preflight potsdam-external-v2-execution-preflight potsdam-external-acceptance release-train-2-validation-smoke release-train-3-spatial-foundation-smoke release-train-3-mesh-smoke release-train-3-export-smoke release-train-3-workstation-smoke release-train-3-acceptance release-train-4-analytical-smoke sidecar-build release-train-5-sidecar-smoke standalone-build demo-rdsm demo-mesh demo-ui demo-india-absolute benchmark-ortholoc-demo rdah-setup benchmark-rdah-ortholoc rdah-sweep-setup benchmark-rdah-sweep

test:
	python -m pytest

verify:
	python scripts/verify.py

service:
	depthwizard serve --host 127.0.0.1 --port 8765

frontend-build:
	cd apps/desktop && npm install --no-audit --no-fund && npm run build

rust-verify:
	cd apps/desktop/src-tauri && cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-targets --all-features

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

sidecar-build:
	python -m pip install -e ".[standalone]"
	python -m scripts.build_standalone_sidecar

release-train-5-sidecar-smoke:
	python -m scripts.release_train_5_sidecar_smoke

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
