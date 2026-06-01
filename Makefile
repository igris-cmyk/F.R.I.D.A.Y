dev:
	./scripts/dev.sh

health:
	./scripts/health.sh

stop:
	./scripts/stop-dev.sh

eval:
	core/.venv/bin/python -m core.tools.eval_harness run

eval-json:
	core/.venv/bin/python -m core.tools.eval_harness run --json
