#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${script_dir}/lib.sh"

cd "${repo_root}"

pid_file="${pid_dir}/core.pid"
log_file="${log_dir}/core.log"
python_bin="core/.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  missing "Python venv missing at ${python_bin}"
  exit 1
fi

remove_stale_pid "${pid_file}"
if pid_file_is_running "${pid_file}"; then
  ok "Python core already running"
  exit 0
fi

info "Starting Python core..."
nohup "${python_bin}" -m core.main >"${log_file}" 2>&1 &
echo "$!" >"${pid_file}"
sleep 1

if pid_file_is_running "${pid_file}"; then
  ok "Python core started"
else
  rm -f "${pid_file}"
  fail "Python core exited during startup. See ${log_file}"
  exit 1
fi
