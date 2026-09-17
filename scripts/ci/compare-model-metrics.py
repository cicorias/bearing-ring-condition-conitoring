#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

METRIC_NAMES = {"accuracy", "macro_f1", "weighted_f1"}
EPSILON = 1e-12


def load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def collect_metrics(value: Any, path: tuple[str, ...] = ()) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not isinstance(value, dict):
        return metrics

    if value.keys() >= METRIC_NAMES:
        for name in sorted(METRIC_NAMES):
            child = value[name]
            if not isinstance(child, int | float):
                raise ValueError(f"Expected a number at {'.'.join((*path, name))}")
            metric = float(child)
            if not math.isfinite(metric):
                raise ValueError(f"Non-finite metric at {'.'.join((*path, name))}")
            metrics[".".join((*path, name))] = metric

    for key, child in value.items():
        if isinstance(child, dict):
            metrics.update(collect_metrics(child, (*path, key)))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail when tracked model metrics regress from a baseline"
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    baseline = collect_metrics(load_metrics(args.baseline))
    candidate = collect_metrics(load_metrics(args.candidate))
    if not baseline:
        raise ValueError(f"No tracked metrics found in {args.baseline}")

    missing = sorted(set(baseline) - set(candidate))
    extra = sorted(set(candidate) - set(baseline))
    rows: list[tuple[str, float, float, float, str]] = []
    regressions: list[str] = []

    for name in sorted(set(baseline) & set(candidate)):
        before = baseline[name]
        after = candidate[name]
        delta = after - before
        status = "REGRESSION" if delta < -EPSILON else "OK"
        rows.append((name, before, after, delta, status))
        if status == "REGRESSION":
            regressions.append(f"{name}: {before:.6f} -> {after:.6f} ({delta:+.6f})")

    lines = [
        "## Model metric drift",
        "",
        "| Metric | Main baseline | Candidate | Delta | Status |",
        "|---|---:|---:|---:|---|",
        *[
            f"| `{name}` | {before:.6f} | {after:.6f} | {delta:+.6f} | {status} |"
            for name, before, after, delta, status in rows
        ],
    ]
    if missing:
        lines.extend(["", f"Missing candidate metrics: `{', '.join(missing)}`"])
    if extra:
        lines.extend(["", f"New candidate metrics: `{', '.join(extra)}`"])
    report = "\n".join(lines) + "\n"
    print(report)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(report, encoding="utf-8")

    if missing:
        print("Candidate metrics are incomplete.", flush=True)
        return 1
    if regressions:
        print("Model metric regressions detected:", flush=True)
        for regression in regressions:
            print(f"- {regression}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
