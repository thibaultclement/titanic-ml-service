from fastapi import FastAPI

from src.titanic.api.routes.explain import router as explain_router
from src.titanic.api.routes.health import router as health_router
from src.titanic.api.routes.model_info import router as model_info_router
from src.titanic.api.routes.predict import router as predict_router
from src.titanic.api.routes.what_if import router as what_if_router

app = FastAPI(
    title="Titanic ML Service API",
    version="1.0.0",
    description=(
        "API de prédiction de survie Titanic avec explicabilité et what-if analysis."
    ),
)

app.include_router(health_router, tags=["Health"])
app.include_router(model_info_router, tags=["Model"])
app.include_router(predict_router, tags=["Prediction"])
app.include_router(what_if_router, tags=["What-if"])
app.include_router(explain_router, tags=["Explainability"])
