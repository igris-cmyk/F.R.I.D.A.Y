.PHONY: dev stop health eval eval-json test

dev:
	./scripts/dev.sh

stop:
	./scripts/stop-dev.sh

health:
	./scripts/health.sh

eval:
	core/.venv/bin/python -m core.tools.eval_harness run

eval-json:
	core/.venv/bin/python -m core.tools.eval_harness run --json

test:
	core/.venv/bin/python -m unittest discover -s tests
