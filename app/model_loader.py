import joblib
from app.config import MODEL_URI, MLFLOW_TRACKING_URI
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

_model = None
_model_info = None


def load_model():
    global _model
    if _model is None:
        #_model = joblib.load(LOCAL_MODEL_PATH)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _model = mlflow.sklearn.load_model(MODEL_URI)
    return _model

def get_model_info():
    global _model_info
    if _model_info is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            info = mlflow.models.get_model_info(MODEL_URI)
            run = MlflowClient().get_run(info.run_id)

            _model_info = {
                "run_id": info.run_id,
                "model_type": run.data.params.get("model_type"),
                "test_accuracy": run.data.metrics.get("test_accuracy"),
            }
        except Exception:
            _model_info = {
                "run_id": "unknown",
                "model_type": None,
                "test_accuracy": None,
            }
    return _model_info