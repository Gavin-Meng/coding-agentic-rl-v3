#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec .venvs/control/bin/python -m carl_v3.archive "$@"
