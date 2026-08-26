from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from grinder_diagnostics_model.constants import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_FEATURE_PATH,
)
from grinder_diagnostics_model.data import write_manifest
from grinder_diagnostics_model.features import build_feature_table
from grinder_diagnostics_model.inference import InferenceEngine
from grinder_diagnostics_model.settings import ModelSettings
from grinder_diagnostics_model.training import train_and_export


def data_check() -> None:
    settings = ModelSettings()
    parser = argparse.ArgumentParser(description="Validate the complete source dataset")
    parser.add_argument("--data-root", type=Path, default=settings.data_root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/grinder-diagnostics-model/source-manifest.json"),
    )
    args = parser.parse_args()
    summary = write_manifest(args.data_root, args.manifest)
    print(summary.to_json())


def build_features() -> None:
    settings = ModelSettings()
    parser = argparse.ArgumentParser(description="Build idle-segment ring features")
    parser.add_argument("--data-root", type=Path, default=settings.data_root)
    parser.add_argument("--output", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    frame = build_feature_table(args.data_root, args.output, limit=args.limit)
    print(f"wrote {len(frame)} rows and {len(frame.columns)} columns to {args.output}")


def train() -> None:
    parser = argparse.ArgumentParser(description="Train and export both random forests")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    result = train_and_export(args.features, args.artifact_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


def predict() -> None:
    parser = argparse.ArgumentParser(description="Predict from a feature JSON file")
    parser.add_argument("request", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "model.pt",
    )
    args = parser.parse_args()
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    engine = InferenceEngine.load(args.model)
    prediction = engine.predict(payload["features"])
    print(
        json.dumps(
            {"request_id": payload["request_id"], **prediction.__dict__},
            indent=2,
            sort_keys=True,
        )
    )


def sample_client() -> None:
    parser = argparse.ArgumentParser(description="Send a sample prediction API request")
    parser.add_argument(
        "--request",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "sample-request.json",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/predict")
    args = parser.parse_args()
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    response = httpx.post(args.url, json=payload, timeout=30)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, sort_keys=True))
