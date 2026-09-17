#!/usr/bin/env bash
set -euo pipefail

destination="${1:-.baseline}"
workflow="${MODEL_BASELINE_WORKFLOW:-model-refresh.yml}"
branch="${MODEL_BASELINE_BRANCH:-main}"
artifact="${MODEL_BASELINE_ARTIFACT:-model-baseline}"

if ! command -v gh >/dev/null 2>&1; then
  echo "The GitHub CLI is required to download the model baseline." >&2
  exit 1
fi
if [[ -z "${GH_REPO:-}" ]]; then
  echo "GH_REPO must identify the repository containing the baseline." >&2
  exit 1
fi

run_id="$(
  gh run list \
    --repo "$GH_REPO" \
    --workflow "$workflow" \
    --branch "$branch" \
    --status success \
    --limit 1 \
    --json databaseId \
    --jq '.[0].databaseId'
)"
if [[ -z "$run_id" ]]; then
  echo "No successful $workflow run found on $branch." >&2
  exit 2
fi

if [[ -e "$destination" ]]; then
  echo "Baseline destination already exists: $destination" >&2
  exit 1
fi
mkdir -p "$destination"
gh run download \
  "$run_id" \
  --repo "$GH_REPO" \
  --name "$artifact" \
  --dir "$destination"
echo "Downloaded $artifact from workflow run $run_id."
