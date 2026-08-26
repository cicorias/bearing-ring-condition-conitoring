# 02 - Grinder diagnostics model: running and results

## Current result

The complete 735-ring dataset was processed into 160 candidate signal features. Training selects 58
features and fits the two-stage random-forest pipeline:

1. normal versus fault;
2. one of the five fault types when a fault is present.

| Split | Binary accuracy | Five-class accuracy |
|---|---:|---:|
| Paper-like stratified 70/30 | 98.64% | 100.00% |
| Dressing-cycle grouped | 99.05% | 98.00% |

The paper reports binary F1 of 99.54% and five-class global F1 of 99.68%. Our feature extraction is
a close reconstruction, not an exact MATLAB port: the paper does not publish every filter,
segmentation, NCA, or random-seed detail.

### Reduced sensor experiments

| Split and sensors | Binary accuracy | Five-class accuracy |
|---|---:|---:|
| Paper-like, process control | 97.74% | 99.37% |
| Paper-like, condition monitoring | 98.64% | 98.73% |
| Grouped, process control | 97.62% | 100.00% |
| Grouped, condition monitoring | 88.57% | 91.33% |

Full metrics, confusion matrices, class reports, and split membership are under
`artifacts/grinder-diagnostics-model/`.

## Environment

All commands use `mise` and `uv`; none use system Python.

```bash
mise run a:setup
mise run b:model:data:check
mise run c:model:features:build
mise run d:model:train
mise run k:verify
```

The managed Microsoft Python feed comes from `~/.config/uv/uv.toml`. PyTorch uses the explicit
CPU-only index in `pyproject.toml`.

## Generated model files

| File | Purpose |
|---|---|
| `model.pt` | PyTorch-loadable two-forest inference artifact |
| `metadata.json` | Feature order, labels, threshold, version, and provenance |
| `sample-request.json` | Valid request generated from an evaluation-split row |
| `metrics.json` | Evaluation results |
| `split-manifest.json` | Exact train/test ring IDs |
| `feature-ranking.json` | Feature-selection output |
| `reference-models.joblib` | scikit-learn reference used only to verify export parity |

The PyTorch and reference forests have a maximum observed probability difference of `0.0`.
The production artifact is retrained on all 735 rings after evaluation, so its sample request is a
format and inference demonstration rather than an additional held-out score.

## Command-line inference

```bash
./scripts/model/predict.sh
```

To use another request:

```bash
./scripts/model/predict.sh path/to/request.json
```

## HTTP inference

Start the API:

```bash
./scripts/model/serve-api.sh
```

In another terminal:

```bash
./scripts/model/sample-client.sh
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Readiness and loaded model version |
| `GET` | `/v1/model` | Required features, labels, and threshold |
| `POST` | `/v1/predict` | Two-stage prediction |

Request shape:

```json
{
  "request_id": "caller-generated-id",
  "features": {
    "exact_feature_name": 1.23
  }
}
```

Use `GET /v1/model` for the complete feature schema or start from
`artifacts/grinder-diagnostics-model/sample-request.json`.

The response includes normal/fault probability, fault type and probabilities when applicable, model
and request identifiers, provenance, and `downstream_analysis_required`. A future generative-AI
pipeline can branch on that boolean and consume the structured evidence without loading the model or
depending on training code.

## HTTPYac contract tests

With the API running:

```bash
mise run l:http:setup
mise run m:http:test
```

The request files are under `tests/http/` and cover health, model metadata, sample prediction, and
invalid feature-schema handling.
