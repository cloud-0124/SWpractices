import os

MODEL_MODE = "ml" # "rules"
# LOCAL_MODEL_PATH = "ml/artifacts/spam_model.joblib"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
# MODEL_URI = "runs:/f40bca671bc64b77b95fdc89cb320aff/model"
MODEL_URI = "models:/spam-model@champion"