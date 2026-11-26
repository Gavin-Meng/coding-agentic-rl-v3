#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
python="$CARL_V3_ROOT/.venvs/control/bin/python"
if [[ "${1:-}" == "--tasks-only" || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "$python" -m carl_v3.prepare "$@"
fi
"$python" -c 'from carl_v3.preflight import local_docker; print(local_docker())'
mkdir -p .state
exec 7>.state/assets.lock
flock -n 7
exec "$python" -m carl_v3.prepare "$@"
