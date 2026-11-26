#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec python3 -m carl_v3.preflight "$@"
