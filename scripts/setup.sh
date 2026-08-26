#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/_common.sh"
"$MISE_BIN" exec -- uv sync --locked
