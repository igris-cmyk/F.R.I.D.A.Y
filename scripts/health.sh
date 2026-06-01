#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${script_dir}/lib.sh"

cd "${repo_root}"

status=0
ollama_url="${OLLAMA_BASE_URL:-http://localhost:11434}"
tags_url="${ollama_url%/}/api/tags"
required_models=(
  "qwen2.5:1.5b"
  "qwen2.5:3b"
  "qwen2.5-coder:1.5b"
  "nomic-embed-text"
)

tags_payload=""
if tags_payload="$(fetch_url "${tags_url}" 2>/dev/null)"; then
  ok "Ollama reachable"
else
  missing "Ollama not reachable at ${tags_url}"
  printf 'Run: ollama serve\n'
  status=1
fi

installed_models=""
if command -v ollama >/dev/null 2>&1; then
  installed_models="$(ollama list 2>/dev/null | awk 'NR > 1 && $1 != "" {print $1}' || true)"
fi

if [[ -n "${tags_payload}" ]] && command -v python3 >/dev/null 2>&1; then
  tags_models="$(
    python3 -c '
import json
import sys

try:
    payload = json.loads(sys.stdin.read())
except Exception:
    raise SystemExit(0)

for model in payload.get("models", []):
    if isinstance(model, dict):
        name = model.get("model") or model.get("name")
        if name:
            print(name)
' <<<"${tags_payload}"
  )"
  installed_models="$(
    printf '%s\n%s\n' "${installed_models}" "${tags_models}" |
      awk 'NF && !seen[$0]++'
  )"
fi

for model in "${required_models[@]}"; do
  if model_list_contains "${model}" "${installed_models}"; then
    ok "${model} installed"
  else
    missing "${model}"
    printf 'Run: ollama pull %s\n' "${model}"
    status=1
  fi
done

if port_listening 9222; then
  ok "NATS WebSocket listening on 9222"
else
  missing "NATS WebSocket not listening on 9222"
  printf 'Run: ./scripts/start-nats.sh\n'
  status=1
fi

nats_tcp_port="$(awk '/^[[:space:]]*port:[[:space:]]*[0-9]+/ {print $2; exit}' infra/nats.conf 2>/dev/null || true)"
if [[ -n "${nats_tcp_port}" ]]; then
  if port_listening "${nats_tcp_port}"; then
    ok "NATS TCP listening on ${nats_tcp_port}"
  else
    missing "NATS TCP not listening on ${nats_tcp_port}"
    printf 'Run: ./scripts/start-nats.sh\n'
    status=1
  fi
else
  missing "NATS TCP port not found in infra/nats.conf"
  status=1
fi

if [[ -x core/.venv/bin/python ]]; then
  ok "Python venv found"
else
  missing "Python venv missing at core/.venv/bin/python"
  printf 'Create/install the core virtual environment before starting FRIDAY.\n'
  status=1
fi

if [[ -f apps/desktop/package.json ]]; then
  ok "Desktop package found"
else
  missing "Desktop package missing at apps/desktop/package.json"
  status=1
fi

memory_db_path="${FRIDAY_MEMORY_DB_PATH:-.friday/memory/friday_memory.sqlite3}"
memory_db_dir="$(dirname "${memory_db_path}")"
if mkdir -p "${memory_db_dir}" 2>/dev/null && [[ -w "${memory_db_dir}" ]]; then
  ok "Memory DB path writable: ${memory_db_path}"
else
  missing "Memory DB path not writable: ${memory_db_path}"
  status=1
fi

if [[ -x core/.venv/bin/python ]] && command -v python3 >/dev/null 2>&1; then
  memory_health_json="$(core/.venv/bin/python -m core.tools.memory_debug health 2>/dev/null || true)"
  memory_schema_version="$(
    python3 -c '
import json
import sys

try:
    payload = json.loads(sys.stdin.read())
except Exception:
    raise SystemExit(0)

schema_version = payload.get("schema_version")
target = payload.get("target_schema_version")
status = payload.get("migration_status")
if schema_version is not None and target is not None and status:
    print(f"{schema_version}|{target}|{status}")
' <<<"${memory_health_json}"
  )"
  if [[ -n "${memory_schema_version}" ]]; then
    IFS='|' read -r schema_version target_schema_version migration_status <<<"${memory_schema_version}"
    if [[ "${migration_status}" == "ok" ]]; then
      ok "Memory schema version: ${schema_version}/${target_schema_version}"
    else
      missing "Memory schema migration status: ${migration_status}"
      printf '%s\n' "${memory_health_json}"
      status=1
    fi
  else
    missing "Memory schema version unavailable"
  fi
fi

exit "${status}"
