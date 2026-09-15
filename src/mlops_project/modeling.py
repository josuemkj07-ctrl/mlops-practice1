import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(features: pd.DataFrame, task: str, model_params: dict) -> Pipeline:
    """Construit un pipeline reproductible : imputation, encodage puis modèle."""
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ],
        remainder="drop",
    )
    parameters = {key: value for key, value in model_params.items() if key != "name"}
    if task == "classification":
        estimator = RandomForestClassifier(random_state=42, **parameters)
    elif task == "regression":
        estimator = RandomForestRegressor(random_state=42, **parameters)
    else:
        raise ValueError("task doit être 'classification' ou 'regression'.")
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])
