import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from grinder_diagnostics_model.constants import DEFAULT_ARTIFACT_DIR
from grinder_diagnostics_model.inference import InferenceEngine


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=200)
    features: dict[str, float]


class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    has_fault: bool
    binary_probability: float
    binary_probabilities: dict[str, float]
    fault_type: str | None
    fault_probabilities: dict[str, float]
    downstream_analysis_required: bool
    warnings: list[str]
    provenance: dict[str, str]


def create_app(engine: InferenceEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if engine is not None:
            app.state.engine = engine
        else:
            default_path = DEFAULT_ARTIFACT_DIR / "model.pt"
            artifact_path = Path(os.getenv("GRINDER_DIAGNOSTICS_MODEL_PATH", default_path))
            app.state.engine = InferenceEngine.load(artifact_path)
        yield

    application = FastAPI(
        title="Grinder Diagnostics Model API",
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_engine(request: Request) -> InferenceEngine:
        return request.app.state.engine

    @application.get("/health")
    def health(model: Annotated[InferenceEngine, Depends(get_engine)]) -> dict[str, str]:
        return {"status": "ok", "model_version": str(model.metadata["model_version"])}

    @application.get("/v1/model")
    def model_info(model: Annotated[InferenceEngine, Depends(get_engine)]) -> dict[str, object]:
        return {
            "model_version": model.metadata["model_version"],
            "feature_count": len(model.feature_names),
            "feature_names": model.feature_names,
            "binary_threshold": model.metadata["binary_threshold"],
            "fault_labels": model.metadata["fault_labels"],
        }

    @application.post("/v1/predict", response_model=PredictionResponse)
    def predict(
        payload: PredictionRequest,
        model: Annotated[InferenceEngine, Depends(get_engine)],
    ) -> PredictionResponse:
        try:
            prediction = model.predict(payload.features)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return PredictionResponse(
            request_id=payload.request_id,
            **prediction.__dict__,
        )

    return application


app = create_app()
