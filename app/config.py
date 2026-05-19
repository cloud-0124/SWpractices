import os

MODEL_MODE = "ml" # "rules" 
# MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_TRACKING_URI = "https://fling-diminish-unstylish.ngrok-free.dev"
MODEL_URI = "models:/spam-model@challenger"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

MODEL_NAME = "spam_model.joblib"

ARTIFACT_DIR_NAME = "artifacts"
DATA_DIR_NAME = "data"