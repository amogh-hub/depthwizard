.PHONY: test verify service frontend-build da3-setup da3-smoke height-model-smoke height-train-acceptance height-multiscene-v2 height-multiscene-v3 height-multiscene-v4 height-multiscene-acceptance height-adaptive-acceptance height-frozen-holdout-v1 height-frozen-holdout potsdam-contract-audit potsdam-external-v1 potsdam-external-v2-preflight potsdam-external-acceptance demo-rdsm demo-mesh demo-ui demo-india-absolute benchmark-ortholoc-demo rdah-setup benchmark-rdah-ortholoc rdah-sweep-setup benchmark-rdah-sweep

test:
	python -m pytest

verify:
	python scripts/verify.py

service:
	depthwizard serve --host 127.0.0.1 --port 8765

frontend-build:
	cd apps/desktop && npm install --no-audit --no-fund && npm run build

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

potsdam-external-acceptance:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python -m scripts.evaluate_potsdam_external_v2

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
