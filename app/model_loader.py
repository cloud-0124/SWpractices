import random

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from app.config import (
    CANARY_ENABLED,
    CANARY_RATIO,
    CHALLENGER_MODEL_URI,
    CHAMPION_MODEL_URI,
    MLFLOW_TRACKING_URI,
    MODEL_URI,
)

_model = None
_champion_model = None
_challenger_model = None
_model_info_cache = {}


def load_model():
    global _model
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _model = mlflow.sklearn.load_model(MODEL_URI)
    return _model


def load_champion_model():
    global _champion_model
    if _champion_model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _champion_model = mlflow.sklearn.load_model(CHAMPION_MODEL_URI)
    return _champion_model


def load_challenger_model():
    global _challenger_model
    if _challenger_model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _challenger_model = mlflow.sklearn.load_model(CHALLENGER_MODEL_URI)
    return _challenger_model


def select_serving_model():
    if CANARY_ENABLED and random.random() < CANARY_RATIO:
        return load_challenger_model(), "challenger"
    return load_champion_model(), "champion"


def get_model_info(serving_model: str = "default"):
    if serving_model in _model_info_cache:
        return _model_info_cache[serving_model]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        if serving_model == "champion":
            info = mlflow.models.get_model_info(CHAMPION_MODEL_URI)
        elif serving_model == "challenger":
            info = mlflow.models.get_model_info(CHALLENGER_MODEL_URI)
        else:
            info = mlflow.models.get_model_info(MODEL_URI)

        run = MlflowClient().get_run(info.run_id)
        _model_info_cache[serving_model] = {
            "run_id": info.run_id,
            "model_type": run.data.params.get("model_type"),
            "test_accuracy": run.data.metrics.get("test_accuracy"),
        }
    except Exception:
        _model_info_cache[serving_model] = {
            "run_id": "unknown",
            "model_type": None,
            "test_accuracy": None,
        }

    return _model_info_cache[serving_model]
