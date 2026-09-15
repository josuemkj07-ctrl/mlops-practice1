from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Charge la configuration YAML versionnée du projet."""
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)
