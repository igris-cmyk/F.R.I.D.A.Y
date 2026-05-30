#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${script_dir}/lib.sh"

cd "${repo_root}"

log_file="${log_dir}/desktop.log"

if ! command -v npm >/dev/null 2>&1; then
  missing "npm not found. Install Node.js/npm first."
  exit 1
fi

if [[ ! -f apps/desktop/package.json ]]; then
  missing "Desktop package missing at apps/desktop/package.json"
  exit 1
fi

info "Starting Tauri desktop dev app..."
info "Desktop log: ${log_file}"
cd apps/desktop
npm run tauri dev 2>&1 | tee "${log_file}"
