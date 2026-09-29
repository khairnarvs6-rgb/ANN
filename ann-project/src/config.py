from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/config.yaml"):
    """Load YAML config. Paths inside are resolved relative to project root."""
    with open(PROJECT_ROOT / path, "r") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = str(PROJECT_ROOT / v)
    return cfg
