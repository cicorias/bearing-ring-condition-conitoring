#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../_common.sh"
export GRINDER_DIAGNOSTICS_MODEL_PATH="${GRINDER_DIAGNOSTICS_MODEL_PATH:-artifacts/grinder-diagnostics-model/model.pt}"
"$MISE_BIN" exec -- uv run uvicorn grinder_diagnostics_model.api:app --host 127.0.0.1 --port 8000 "$@"
