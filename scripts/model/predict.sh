#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../_common.sh"
request_path="${1:-artifacts/grinder-diagnostics-model/sample-request.json}"
shift || true
"$MISE_BIN" exec -- uv run grinder-model-predict "$request_path" "$@"
