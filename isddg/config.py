from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _looks_like_file_path(path: Path, key: str = "") -> bool:
    """Heuristically decide whether a configured path is a file path.

    The previous implementation created every value under cfg["paths"] as a
    directory. That is unsafe because keys such as "prototype_bank" or
    "prototype_summary" point to files like *.pt / *.json. Creating those as
    directories later makes torch.save/open fail.

    Directory keys should either end with "_dir" or have no file suffix.
    File keys usually have suffixes such as .pt, .json, .csv, .yaml, etc.
    """
    key_l = key.lower()
    if key_l.endswith("_dir") or key_l in {"result_dir", "results_dir", "out_dir", "output_dir", "log_dir", "logs_dir"}:
        return False
    if path.suffix:
        return True
    return False


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """Create output directories declared in config.

    For a directory path, create the directory itself.
    For a file path, create only its parent directory.

    This keeps configs flexible while avoiding accidental folders named
    "xxx.pt", "xxx.json", or "xxx.csv".
    """
    for key, value in cfg.get("paths", {}).items():
        if value is None or value == "":
            continue
        path = Path(value)
        if _looks_like_file_path(path, key):
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    # Some experiment-specific sections also contain output file paths.
    for section_name in ("prototype", "semantic", "continuous_dynamic", "dynamic", "roles"):
        section = cfg.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if value is None or value == "":
                continue
            if not isinstance(value, (str, Path)):
                continue
            path = Path(value)
            key_l = key.lower()
            if key_l.endswith(("path", "file", "checkpoint", "summary", "csv", "json", "pt")) or path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
