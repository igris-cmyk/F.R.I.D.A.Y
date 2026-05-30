#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${script_dir}/lib.sh"

cd "${repo_root}"

shopt -s nullglob
pid_files=("${pid_dir}"/*.pid)

if (( ${#pid_files[@]} == 0 )); then
  info "No FRIDAY script-managed PID files found."
  exit 0
fi

for pid_file in "${pid_files[@]}"; do
  service="$(basename "${pid_file}" .pid)"
  pid="$(cat "${pid_file}" 2>/dev/null || true)"

  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
    info "Removed empty PID file for ${service}"
    continue
  fi

  if ! pid_is_running "${pid}"; then
    rm -f "${pid_file}"
    info "Removed stale PID for ${service}"
    continue
  fi

  kill "${pid}"
  for _ in $(seq 1 20); do
    if ! pid_is_running "${pid}"; then
      break
    fi
    sleep 0.2
  done

  if pid_is_running "${pid}"; then
    kill -TERM "${pid}" 2>/dev/null || true
  fi

  rm -f "${pid_file}"
  ok "Stopped ${service} pid=${pid}"
done
