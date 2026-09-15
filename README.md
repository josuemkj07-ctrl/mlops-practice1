# Projet MLOps collaboratif

Ce dépôt sert de socle à un projet de machine learning reproductible à trois : préparation de données, analyse descriptive, entraînement d'un modèle de classification ou de régression, évaluation et suivi avec MLflow.

## Démarrage

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

1. Déposez le jeu de données non versionné dans `data/raw/dataset.csv`.
2. Renseignez sa cible et le type de problème dans `configs/base.yaml`.
3. Exécutez `python -m mlops_project.train --config configs/base.yaml`.
4. Consultez les expériences : `mlflow ui --backend-store-uri ./mlruns`.

Le notebook final est `notebooks/03_final_model.ipynb`. Il n'implémente aucune logique métier : il importe seulement les fonctions de `src/` et centralise les paramètres documentés.

## Répartition proposée

| Rôle | Zone de responsabilité |
| --- | --- |
| Membre 1 | acquisition/qualité des données, EDA et `src/.../data.py` |
| Membre 2 | features, modèles et `src/.../modeling.py` |
| Membre 3 | MLflow, évaluation, tests, CI et documentation |

Chaque zone reste revue par au moins une autre personne via pull request.

## Règles Git à trois

- `main` est protégée : aucun commit direct, uniquement une pull request validée par un pair et une CI verte.
- Une tâche = une branche courte : `feature/nom-court`, `fix/nom-court` ou `docs/nom-court`.
- Avant de commencer et juste avant une PR : `git switch main`, `git pull --ff-only origin main`, puis créez/rebasez votre branche.
- Faites des commits petits et explicites : `feat: add baseline classifier`, `test: cover data validation`.
- Une PR ne mélange pas notebook, refactorisation et nouvelle fonctionnalité. Ne versionnez jamais données brutes, environnements ou exécutions MLflow.
- Résolvez les conflits sur votre branche, testez, puis demandez une revue. Utilisez le squash merge pour garder `main` lisible.

## Architecture

```text
configs/        paramètres versionnés
data/           données locales ignorées par Git
notebooks/      exploration et notebook final d'orchestration
src/            code métier testable et réutilisable
tests/          tests automatisés
reports/        figures et livrables légers
.github/        intégration continue
```
