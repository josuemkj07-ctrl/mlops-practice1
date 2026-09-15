from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred, task: str) -> dict[str, float]:
    """Calcule les métriques principales selon le type de problème."""
    if task == "classification":
        return {"accuracy": accuracy_score(y_true, y_pred), "f1_weighted": f1_score(y_true, y_pred, average="weighted")}
    return {"mae": mean_absolute_error(y_true, y_pred), "r2": r2_score(y_true, y_pred)}
