#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
friday_dir="${repo_root}/.friday"
log_dir="${friday_dir}/logs"
pid_dir="${friday_dir}/pids"

mkdir -p "${log_dir}" "${pid_dir}"

ok() {
  printf '[OK] %s\n' "$*"
}

missing() {
  printf '[MISSING] %s\n' "$*"
}

info() {
  printf '[INFO] %s\n' "$*"
}

fail() {
  printf '[ERROR] %s\n' "$*" >&2
}

pid_is_running() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

pid_file_is_running() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  pid_is_running "${pid}"
}

remove_stale_pid() {
  local pid_file="$1"
  if [[ -f "${pid_file}" ]] && ! pid_file_is_running "${pid_file}"; then
    rm -f "${pid_file}"
  fi
}

http_reachable() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 2 "${url}" >/dev/null 2>&1
    return $?
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$url" >/dev/null 2>&1 <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=2):
        pass
except Exception:
    raise SystemExit(1)
PY
    return $?
  fi

  return 1
}

fetch_url() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 2 "${url}"
    return $?
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$url" <<'PY'
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    print(response.read().decode("utf-8"))
PY
    return $?
  fi

  return 1
}

model_matches_requirement() {
  local required="$1"
  local installed="$2"

  if [[ "${required}" == *:* ]]; then
    [[ "${installed}" == "${required}" ]]
  else
    [[ "${installed}" == "${required}" || "${installed}" == "${required}:latest" ]]
  fi
}

model_list_contains() {
  local required="$1"
  local installed_models="$2"
  local installed

  while IFS= read -r installed; do
    [[ -n "${installed}" ]] || continue
    if model_matches_requirement "${required}" "${installed}"; then
      return 0
    fi
  done <<<"${installed_models}"

  return 1
}

port_listening() {
  local port="$1"

  if command -v ss >/dev/null 2>&1; then
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(:|\\])${port}$"; then
      return 0
    fi
  fi

  if command -v nc >/dev/null 2>&1; then
    if nc -z 127.0.0.1 "${port}" >/dev/null 2>&1; then
      return 0
    fi
  fi

  timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1
}

wait_for_http() {
  local url="$1"
  local attempts="${2:-20}"
  local delay="${3:-0.25}"

  for _ in $(seq 1 "${attempts}"); do
    if http_reachable "${url}"; then
      return 0
    fi
    sleep "${delay}"
  done
  return 1
}

wait_for_port() {
  local port="$1"
  local attempts="${2:-20}"
  local delay="${3:-0.25}"

  for _ in $(seq 1 "${attempts}"); do
    if port_listening "${port}"; then
      return 0
    fi
    sleep "${delay}"
  done
  return 1
}
