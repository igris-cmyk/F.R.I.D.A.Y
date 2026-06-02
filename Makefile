.PHONY: dev stop health eval eval-json eval-security eval-all test index index-status index-search

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

eval-security:
	core/.venv/bin/python -m core.tools.eval_harness run --suite security

eval-all:
	core/.venv/bin/python -m core.tools.eval_harness run --suite all

test:
	core/.venv/bin/python -m unittest discover -s tests

index:
	core/.venv/bin/python -m core.tools.workspace_index build

index-status:
	core/.venv/bin/python -m core.tools.workspace_index status

index-search:
	core/.venv/bin/python -m core.tools.workspace_index search "$(q)"
