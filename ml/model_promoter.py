import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "spam-model-new"


def get_champion_test_accuracy(client: MlflowClient) -> float:
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_run = client.get_run(champion.run_id)
        return champion_run.data.metrics.get("test_accuracy", -1.0)
    except Exception:
        return -1.0


def promote_if_better(new_version: str, new_test_accuracy: float):
    client = MlflowClient()
    current_champion_acc = get_champion_test_accuracy(client)
    print(f"[PROMOTION] current champion test_accuracy = {current_champion_acc}")
    print(f"[PROMOTION] new candidate test_accuracy = {new_test_accuracy}")

    if new_test_accuracy > current_champion_acc:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=str(new_version)
        )
        print(f"[PROMOTION] version {new_version} promoted to champion")
    else:
        print("[PROMOTION] champion unchanged")
