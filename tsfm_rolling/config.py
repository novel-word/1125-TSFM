from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ("data", "forecast", "model", "output"):
        if key not in cfg:
            raise KeyError(f"Missing config section: {key}")
    return cfg
