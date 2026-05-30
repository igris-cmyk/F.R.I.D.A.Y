#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib.sh
source "${script_dir}/lib.sh"

cd "${repo_root}"
mkdir -p "${log_dir}" "${pid_dir}"

cat <<'BANNER'
F.R.I.D.A.Y Dev Runtime
BANNER

"${script_dir}/start-ollama.sh"
"${script_dir}/start-nats.sh"
"${script_dir}/health.sh"
"${script_dir}/start-core.sh"

cat <<EOF

Press Super + Space to summon FRIDAY.
Logs:
- .friday/logs/ollama.log
- .friday/logs/nats.log
- .friday/logs/core.log
- .friday/logs/desktop.log

Starting desktop in the foreground. Use ./scripts/stop-dev.sh to stop script-managed background services.
EOF

"${script_dir}/start-desktop.sh"
