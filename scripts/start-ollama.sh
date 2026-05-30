#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${script_dir}/lib.sh"

cd "${repo_root}"

ollama_url="${OLLAMA_BASE_URL:-http://localhost:11434}"
tags_url="${ollama_url%/}/api/tags"
pid_file="${pid_dir}/ollama.pid"
log_file="${log_dir}/ollama.log"

if http_reachable "${tags_url}"; then
  ok "Ollama already running"
  exit 0
fi

if ! command -v ollama >/dev/null 2>&1; then
  missing "ollama command not found. Install Ollama first."
  exit 1
fi

remove_stale_pid "${pid_file}"
if pid_file_is_running "${pid_file}"; then
  ok "Ollama already starting via PID $(cat "${pid_file}")"
  exit 0
fi

info "Starting Ollama..."
nohup ollama serve >"${log_file}" 2>&1 &
echo "$!" >"${pid_file}"

if wait_for_http "${tags_url}" 30 0.5; then
  ok "Ollama started"
else
  fail "Ollama did not become reachable. See ${log_file}"
  exit 1
fi
