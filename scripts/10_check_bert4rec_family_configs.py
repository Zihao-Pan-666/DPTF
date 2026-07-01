from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


DEFAULT_CONFIGS = [
    "configs/bert4rec_sem_matched.yaml",
    "configs/bert4rec_recg_matched.yaml",
    "configs/bert4rec_sage_matched.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that Sem/RecG/SAGE share one experimental protocol."
    )
    parser.add_argument("configs", nargs="*", default=DEFAULT_CONFIGS)
    return parser.parse_args()


def canonical_common(cfg: dict) -> dict:
    model = dict(cfg["model"])
    model.pop("architecture", None)
    training = dict(cfg["training"])
    evaluation = dict(cfg["evaluation"])
    return {
        "data_root": cfg["data_root"],
        "source": cfg["source"],
        "targets": cfg["targets"],
        "aux_domains": cfg.get("aux_domains", []),
        "data": cfg["data"],
        "model_without_architecture": model,
        "training": training,
        "evaluation": evaluation,
    }


def main() -> None:
    args = parse_args()
    loaded: list[tuple[str, dict]] = []
    for file_name in args.configs:
        with Path(file_name).open("r", encoding="utf-8") as handle:
            loaded.append((file_name, yaml.safe_load(handle)))

    reference_name, reference_cfg = loaded[0]
    reference = canonical_common(reference_cfg)
    errors: list[str] = []

    for file_name, cfg in loaded:
        mode = str(cfg["experiment"]["mode"]).lower()
        architecture = str(cfg["experiment"]["architecture"]).lower()
        expected = "single" if mode == "sem" else "dual"
        if architecture != expected:
            errors.append(
                f"{file_name}: mode={mode} must use architecture={expected}, "
                f"not {architecture}"
            )
        if canonical_common(cfg) != reference:
            errors.append(
                f"{file_name}: common protocol differs from {reference_name}"
            )

    if errors:
        raise SystemExit("\n".join(errors))

    print("[PASS] Sem, RecG, and SAGE share the same data/model/training/evaluation protocol.")
    print(json.dumps(reference, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
