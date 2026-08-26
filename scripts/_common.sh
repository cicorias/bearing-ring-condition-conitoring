#!/usr/bin/env bash

if command -v mise >/dev/null 2>&1; then
  MISE_BIN="$(command -v mise)"
elif [[ -x "$HOME/.local/bin/mise" ]]; then
  MISE_BIN="$HOME/.local/bin/mise"
else
  echo "mise is required but was not found" >&2
  exit 127
fi
