from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path, target: str) -> pd.DataFrame:
    """Lit un CSV et vérifie la présence de la variable cible."""
    dataset = pd.read_csv(path)
    if target not in dataset.columns:
        raise ValueError(f"La cible '{target}' est absente de {path}.")
    if dataset.empty:
        raise ValueError("Le jeu de données est vide.")
    return dataset


def descriptive_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    """Retourne une synthèse descriptive exploitable dans un notebook ou rapport."""
    return dataset.describe(include="all").T
