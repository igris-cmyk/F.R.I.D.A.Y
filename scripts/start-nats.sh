#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${script_dir}/lib.sh"

cd "${repo_root}"

pid_file="${pid_dir}/nats.pid"
log_file="${log_dir}/nats.log"
nats_bin="./infra/nats-server-v2.10.14-linux-amd64/nats-server"

if port_listening 9222; then
  ok "NATS already running"
  exit 0
fi

if [[ ! -x "${nats_bin}" ]]; then
  missing "NATS binary missing or not executable at ${nats_bin}"
  exit 1
fi

if [[ ! -f infra/nats.conf ]]; then
  missing "NATS config missing at infra/nats.conf"
  exit 1
fi

remove_stale_pid "${pid_file}"
if pid_file_is_running "${pid_file}"; then
  ok "NATS already starting via PID $(cat "${pid_file}")"
  exit 0
fi

info "Starting NATS..."
nohup "${nats_bin}" -c infra/nats.conf >"${log_file}" 2>&1 &
echo "$!" >"${pid_file}"

if wait_for_port 9222 30 0.5; then
  ok "NATS started"
else
  fail "NATS did not open WebSocket port 9222. See ${log_file}"
  exit 1
fi
