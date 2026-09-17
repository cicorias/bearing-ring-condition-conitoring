#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../_common.sh"

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
data_root="${GRINDER_DIAGNOSTICS_DATA_ROOT:-$repo_root/data/source}"
dataset_base_url="https://api.researchdata.se/dataset/2022-136-1/2/file"

usage() {
  cat <<EOF
Usage: $0 [--data-root PATH]

Download and extract the published bearing-ring grinder dataset, then validate it.

Options:
  --data-root PATH  Destination directory (default: GRINDER_DIAGNOSTICS_DATA_ROOT
                    or $repo_root/data/source)
  -h, --help        Show this help
EOF
}

while (($#)); do
  case "$1" in
    --data-root)
      if (($# < 2)); then
        echo "--data-root requires a path" >&2
        exit 2
      fi
      data_root="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$data_root"
data_root="$(cd "$data_root" && pwd)"

download() {
  local relative_path="$1"
  local expected_size="$2"
  local url="$3"
  local destination="$data_root/$relative_path"
  local partial="$destination.part"
  local actual_size

  mkdir -p "$(dirname "$destination")"

  if [[ -f "$destination" ]]; then
    actual_size="$(stat -c %s "$destination")"
    if [[ "$actual_size" == "$expected_size" ]]; then
      echo "Already downloaded: $relative_path"
      return
    fi
    echo "Existing file has the wrong size: $destination" >&2
    echo "Expected $expected_size bytes, found $actual_size bytes" >&2
    exit 1
  fi

  echo "Downloading: $relative_path"
  curl \
    --fail \
    --location \
    --retry 5 \
    --retry-all-errors \
    --continue-at - \
    --output "$partial" \
    "$url"

  actual_size="$(stat -c %s "$partial")"
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "Downloaded file has the wrong size: $partial" >&2
    echo "Expected $expected_size bytes, found $actual_size bytes" >&2
    exit 1
  fi
  mv "$partial" "$destination"
}

extract_test() {
  local test_number="$1"
  local archive="$data_root/test_$test_number.zip"
  local destination="$data_root/test_$test_number"
  local ring_count

  ring_count=0
  if [[ -d "$destination/test_$test_number" ]]; then
    ring_count="$(
      find "$destination/test_$test_number" -type f -name 'ring_*.tdms' | wc -l
    )"
  fi
  if [[ "$ring_count" == "105" ]]; then
    echo "Already extracted: test_$test_number"
    return
  fi

  echo "Checking archive: test_$test_number.zip"
  "$MISE_BIN" exec -- python -m zipfile -t "$archive"

  echo "Extracting: test_$test_number.zip"
  "$MISE_BIN" exec -- python - "$archive" "$destination" <<'PY'
from pathlib import Path
import sys
import zipfile

archive = Path(sys.argv[1])
destination = Path(sys.argv[2]).resolve()
destination.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(archive) as source:
    for member in source.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Unsafe archive path: {member.filename}")
    source.extractall(destination)
PY
}

echo "Dataset destination: $data_root"
echo "The complete download is about 20 GB and extraction requires about 89 GB more."

download \
  "proc_param/proc_param/process_data.csv" \
  "19803" \
  "$dataset_base_url/data?filePath=proc_param%2Fproc_param%2Fprocess_data.csv"
download \
  "quality/quality/measured_quality_param.csv" \
  "32229" \
  "$dataset_base_url/data?filePath=quality%2Fquality%2Fmeasured_quality_param.csv"
download \
  "quality/quality/quality_disposition.csv" \
  "4763" \
  "$dataset_base_url/data?filePath=quality%2Fquality%2Fquality_disposition.csv"

archive_sizes=(
  2287181724
  2329474777
  2292608068
  2293611666
  5846872141
  2718481588
  2274483682
)

for test_number in {1..7}; do
  download \
    "test_$test_number.zip" \
    "${archive_sizes[$((test_number - 1))]}" \
    "$dataset_base_url/data?filePath=test_$test_number.zip"
  extract_test "$test_number"
done

echo "Validating downloaded dataset"
"$(dirname "$0")/data-check.sh" --data-root "$data_root"
