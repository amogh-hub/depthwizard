.PHONY: test verify service frontend-build

test:
	python -m pytest

verify:
	python scripts/verify.py

service:
	depthwizard serve --host 127.0.0.1 --port 8765

frontend-build:
	cd apps/desktop && npm run build
