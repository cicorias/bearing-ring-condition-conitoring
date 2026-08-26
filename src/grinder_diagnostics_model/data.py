from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from grinder_diagnostics_model.constants import TEST_LABELS

RING_PATH_PATTERN = re.compile(
    r"test_(?P<test>[1-7])/test_(?P=test)/dresscyc_(?P<dress>[1-7])/"
    r"ring_(?P<ring>1[0-5]|[1-9])\.tdms$"
)


@dataclass(frozen=True, order=True)
class RingRecord:
    test: int
    dressing_cycle: int
    ring: int
    path: Path

    @property
    def ring_id(self) -> str:
        return f"test_{self.test}:dress_{self.dressing_cycle}:ring_{self.ring}"

    @property
    def condition(self) -> str:
        return TEST_LABELS[self.test]

    @property
    def has_fault(self) -> int:
        return int(self.test not in {1, 7})


@dataclass(frozen=True)
class SourceSummary:
    data_root: str
    tdms_files: int
    tests: dict[int, int]
    zip_tdms_files: dict[int, int]
    process_rows: int
    measured_quality_rows: int
    quality_disposition_rows: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def discover_rings(data_root: Path) -> list[RingRecord]:
    records: list[RingRecord] = []
    for path in data_root.glob("test_*/test_*/dresscyc_*/ring_*.tdms"):
        relative = path.relative_to(data_root).as_posix()
        match = RING_PATH_PATTERN.fullmatch(relative)
        if match is None:
            raise ValueError(f"Unexpected TDMS path: {path}")
        records.append(
            RingRecord(
                test=int(match.group("test")),
                dressing_cycle=int(match.group("dress")),
                ring=int(match.group("ring")),
                path=path,
            )
        )
    return sorted(records)


def _expected_keys() -> set[tuple[int, int, int]]:
    return {
        (test, dressing_cycle, ring)
        for test in range(1, 8)
        for dressing_cycle in range(1, 8)
        for ring in range(1, 16)
    }


def _record_keys(records: list[RingRecord]) -> set[tuple[int, int, int]]:
    return {(record.test, record.dressing_cycle, record.ring) for record in records}


def _csv_keys(frame: pd.DataFrame, dress_column: str) -> set[tuple[int, int, int]]:
    return {
        (int(row.Test), int(getattr(row, dress_column)), int(row.Ring))
        for row in frame.itertuples(index=False)
    }


def _zip_ring_names(archive: Path, test: int) -> set[str]:
    with zipfile.ZipFile(archive) as zipped:
        return {
            name
            for name in zipped.namelist()
            if name.startswith(f"test_{test}/") and name.endswith(".tdms")
        }


def validate_source(data_root: Path) -> SourceSummary:
    data_root = data_root.resolve()
    records = discover_rings(data_root)
    expected = _expected_keys()
    observed = _record_keys(records)
    if observed != expected:
        missing = sorted(expected - observed)[:10]
        extra = sorted(observed - expected)[:10]
        raise ValueError(
            f"TDMS coverage mismatch: expected 735, found {len(observed)}; "
            f"missing={missing}, extra={extra}"
        )

    extracted_names_by_test: dict[int, set[str]] = {test: set() for test in range(1, 8)}
    for record in records:
        extracted_names_by_test[record.test].add(
            f"test_{record.test}/dresscyc_{record.dressing_cycle}/ring_{record.ring}.tdms"
        )

    zip_counts: dict[int, int] = {}
    for test in range(1, 8):
        archive = data_root / f"test_{test}.zip"
        if not archive.is_file():
            raise FileNotFoundError(f"Missing source archive: {archive}")
        archived_names = _zip_ring_names(archive, test)
        if archived_names != extracted_names_by_test[test]:
            raise ValueError(f"Extracted files do not match {archive.name}")
        zip_counts[test] = len(archived_names)

    process_path = data_root / "proc_param/proc_param/process_data.csv"
    measured_path = data_root / "quality/quality/measured_quality_param.csv"
    disposition_path = data_root / "quality/quality/quality_disposition.csv"
    process = pd.read_csv(process_path)
    measured = pd.read_csv(measured_path)
    disposition = pd.read_csv(disposition_path)

    if _csv_keys(process, "Dress") != expected:
        raise ValueError("Process data does not contain exactly one row for every ring")
    measured_keys = _csv_keys(measured, "DressCyc")
    disposition_keys = _csv_keys(disposition, "DressCyc")
    if measured_keys != disposition_keys:
        raise ValueError("Measured-quality and quality-disposition ring keys differ")

    counts = Counter(record.test for record in records)
    return SourceSummary(
        data_root=str(data_root),
        tdms_files=len(records),
        tests=dict(sorted(counts.items())),
        zip_tdms_files=zip_counts,
        process_rows=len(process),
        measured_quality_rows=len(measured),
        quality_disposition_rows=len(disposition),
    )


def write_manifest(data_root: Path, output_path: Path) -> SourceSummary:
    summary = validate_source(data_root)
    records = discover_rings(data_root.resolve())
    payload = {
        "summary": asdict(summary),
        "rings": [
            {
                "ring_id": record.ring_id,
                "test": record.test,
                "condition": record.condition,
                "has_fault": record.has_fault,
                "dressing_cycle": record.dressing_cycle,
                "ring": record.ring,
                "path": str(record.path),
                "size_bytes": record.path.stat().st_size,
            }
            for record in records
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return summary
