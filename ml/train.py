import os

import joblib
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from app.config import *
from ml.model_promoter import promote_if_better

BASE_DIR = os.path.dirname(__file__)
TRAIN_DATA_PATH = os.path.join(BASE_DIR, DATA_DIR_NAME, TRAIN_FILE_NAME)
TEST_DATA_PATH = os.path.join(BASE_DIR, DATA_DIR_NAME, TEST_FILE_NAME)
ARTIFACT_DIR = os.path.join(BASE_DIR, ARTIFACT_DIR_NAME)
MODEL_PATH = os.path.join(ARTIFACT_DIR, MODEL_NAME)

os.makedirs(ARTIFACT_DIR, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("spam-classification-server")

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

X_train = train_df["text"]
y_train = train_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]

models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "NaiveBayes": MultinomialNB(),
    "DecisionTree": RandomForestClassifier(n_estimators=100, random_state=42),
}

client = MlflowClient()
best_test_acc = -1.0
best_version = None

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline([
            ("vectorizer", CountVectorizer()),
            ("classifier", model),
        ])

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("vectorizer", "CountVectorizer")
        mlflow.log_param("train_data_path", TRAIN_DATA_PATH)
        mlflow.log_param("test_data_path", TEST_DATA_PATH)
        mlflow.log_param("train_row_count", len(train_df))
        mlflow.log_param("test_row_count", len(test_df))

        pipeline.fit(X_train, y_train)

        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        joblib.dump(pipeline, MODEL_PATH)

        mlflow.log_artifact(TRAIN_DATA_PATH)
        mlflow.log_artifact(TEST_DATA_PATH)
        mlflow.log_artifact(MODEL_PATH)

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name="spam-model-new",
        )

        latest_versions = client.search_model_versions("name='spam-model-new'")
        latest_version = max(latest_versions, key=lambda v: int(v.version)).version

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_version = latest_version

        print(f"[{model_name}] Model saved to: {MODEL_PATH}")
        print(f"[{model_name}] train_accuracy: {train_acc:.4f}")
        print(f"[{model_name}] test_accuracy: {test_acc:.4f}")

if best_version is not None:
    promote_if_better(best_version, best_test_acc)
