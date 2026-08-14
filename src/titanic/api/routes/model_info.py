from fastapi import APIRouter

from src.titanic.api.services.model_service import model_service

router = APIRouter()


@router.get("/model-info")
def model_info():
    model = model_service.pipeline.named_steps["model"]

    return {
        "project": "Titanic ML Service",
        "model_name": model.__class__.__name__,
        "model_type": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "version": "1.0.0",
        "local_kaggle_score": 0.8014,
        "features": model_service.features,
        "n_features": len(model_service.features),
        "endpoints": [
            "/health",
            "/model-info",
            "/predict",
            "/what-if",
            "/explain",
        ],
    }