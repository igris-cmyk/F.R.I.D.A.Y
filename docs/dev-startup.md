# F.R.I.D.A.Y. Local Dev Startup

Run all dev commands from the repository root:

```bash
cd /home/igris/F.R.I.D.A.Y
```

Use the dev script directly:

```bash
./scripts/dev.sh
```

Equivalent Make targets:

```bash
make dev
make health
make stop
make eval
make eval-security
make eval-all
```

Useful direct commands:

```bash
./scripts/health.sh
./scripts/stop-dev.sh
core/.venv/bin/python -m core.tools.eval_harness run
core/.venv/bin/python -m core.tools.eval_harness run --suite security
core/.venv/bin/python -m core.tools.eval_harness run --suite all
```

`dev.sh` starts or verifies:

- Ollama at `http://localhost:11434`
- NATS via `./infra/nats-server-v2.10.14-linux-amd64/nats-server -c infra/nats.conf`
- Python core via `core/.venv/bin/python -m core.main`
- Desktop via `cd apps/desktop && npm run tauri dev`

Script-managed logs and PID files live under `.friday/`, which is ignored by git.

## Required Ollama Models

Install the local models before starting FRIDAY:

```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
ollama pull qwen2.5-coder:1.5b
ollama pull nomic-embed-text
```

## Health Check

Run:

```bash
./scripts/health.sh
```

The health check validates Ollama reachability, required models, NATS WebSocket port `9222`, the configured NATS TCP port, the Python venv, and the desktop package.

## Stopping Dev Services

Run:

```bash
./scripts/stop-dev.sh
```

This stops only processes with PID files created by these scripts. It does not use broad process matching and will not kill unrelated user processes.
