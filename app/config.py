import os

MODEL_MODE = "ml" # "rules" 
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_URI = "models:/spam-model@champion"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

MODEL_NAME = "spam_model.joblib"

ARTIFACT_DIR_NAME = "artifacts"
DATA_DIR_NAME = "data"