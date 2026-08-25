.PHONY: test verify service frontend-build da3-setup da3-smoke demo-rdsm demo-mesh demo-ui

test:
	python -m pytest

verify:
	python scripts/verify.py

service:
	depthwizard serve --host 127.0.0.1 --port 8765

frontend-build:
	cd apps/desktop && npm run build

da3-setup:
	bash scripts/setup_da3_macos.sh

da3-smoke:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/smoke_da3.py

demo-rdsm:
	PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/demo_geotiff.py

demo-mesh:
	depthwizard mesh-rdsm artifacts/demo/geotiff-rdsm/rdsm.tif data/demo/RGB.byte.tif artifacts/demo/geotiff-mesh --vertical-scale 5000

demo-ui:
	bash scripts/demo_ui.sh
