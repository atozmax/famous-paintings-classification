from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(name):
    path = ROOT / "config" / f"{name}.yaml"
    with path.open() as handle:
        return yaml.safe_load(handle)


def resolve_path(value):
    if value in (None, "", "null"):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path
