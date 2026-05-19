# ml/train.py
from sklearn.metrics import accuracy_score # 성능 지표 저장을 위해
import mlflow.sklearn # mlflow 형태로 저장

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from app.config import *

BASE_DIR = os.path.dirname(__file__)
TRAIN_DATA_PATH = os.path.join(BASE_DIR, DATA_DIR_NAME,TRAIN_FILE_NAME)
TEST_DATA_PATH = os.path.join(BASE_DIR, DATA_DIR_NAME, TEST_FILE_NAME)
ARTIFACT_DIR = os.path.join(BASE_DIR, ARTIFACT_DIR_NAME)
MODEL_PATH = os.path.join(ARTIFACT_DIR, MODEL_NAME)

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 실험 세팅
# mlflow는 sqlite를 기반 DB로 사용
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("spam-classification-local")

train_df= pd.read_csv(TRAIN_DATA_PATH)
test_df= pd.read_csv(TEST_DATA_PATH)

X_train= train_df["text"]
y_train= train_df["label"]

X_test= test_df["text"]
y_test= test_df["label"]

models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "NaiveBayes": MultinomialNB(),
    "DecisionTree": RandomForestClassifier(n_estimators=100, random_state=42)
}    # Naming bug

# 실험 기록 시작
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline([
            ("vectorizer", CountVectorizer()),
            ("classifier", model)
        ])

        # 실험 설정 기록
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_data_path", TRAIN_DATA_PATH)
        mlflow.log_param("test_train_data_path", TEST_DATA_PATH)
        mlflow.log_param("train_row_count", len(train_df))
        mlflow.log_param("test_row_count", len(test_df))

        pipeline.fit(X_train, y_train)

        # 간단한 metric 저장(train/test accuracy)
        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        joblib.dump(pipeline, MODEL_PATH)

        # artifact 기록
        mlflow.log_artifact(TRAIN_DATA_PATH)
        mlflow.log_artifact(TEST_DATA_PATH)
        mlflow.log_artifact(MODEL_PATH)

        # MLflow 모델 형식으로도 저장
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name="spam-model"
        )

        print(f"[{model_name}] Model saved to: {MODEL_PATH}")
        print(f"[{model_name}] train_accuracy: {train_acc:.4f}")
        print(f"[{model_name}] test_accuracy: {test_acc:.4f}")