import argparse

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from mlops_project.config import load_config
from mlops_project.data import load_dataset
from mlops_project.evaluation import calculate_metrics
from mlops_project.modeling import build_pipeline


def train(config_path: str) -> dict[str, float]:
    """Entraîne, journalise le run MLflow et retourne les métriques de test."""
    config = load_config(config_path)
    data_config = config["data"]
    dataset = load_dataset(data_config["path"], data_config["target"])
    features = dataset.drop(columns=data_config["target"])
    target = dataset[data_config["target"]]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=data_config["test_size"], random_state=data_config["random_state"], stratify=target if data_config["task"] == "classification" else None
    )
    pipeline = build_pipeline(x_train, data_config["task"], config["model"])
    mlflow.set_experiment(config["project"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params({"task": data_config["task"], **config["model"]})
        pipeline.fit(x_train, y_train)
        metrics = calculate_metrics(y_test, pipeline.predict(x_test), data_config["task"])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(train(args.config))
