from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

from isddg.features.semantic import load_semantic_embeddings
from isddg.utils.bert4rec_family_compat import (
    build_source_splits_compat,
    group_user_sequences_compat,
    load_interactions_compat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the exact source semantic tensor used by the matched "
            "BERT4Rec baselines without modifying any embedding file."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/bert4rec_sem_matched.yaml",
        help="Matched-family YAML configuration.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help=(
            "Optional training summary JSON. When supplied, the script compares "
            "the duplicate-vector tie prediction with the observed evaluator ties."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. A data-quality path is used by default.",
    )
    parser.add_argument(
        "--simulate-validation-ties",
        action="store_true",
        help=(
            "Reproduce the fixed sampled-negative validation protocol and measure "
            "how many target ties are explained by exact duplicate embeddings."
        ),
    )
    parser.add_argument(
        "--max-duplicate-groups",
        type=int,
        default=20,
        help="Maximum duplicate groups included in the JSON report.",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Show progress bars. auto enables them in an interactive terminal.",
    )
    return parser.parse_args()


def _progress(iterable: Iterable, total: int | None, desc: str, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def _hash_item_map(item_map: dict[str, int]) -> str:
    digest = hashlib.sha256()
    for raw_id, index in sorted(item_map.items(), key=lambda pair: int(pair[1])):
        digest.update(str(index).encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(raw_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _row_digest(row: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(row, dtype=np.float32)
    return hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest()


def _tensor_fingerprint(matrix: np.ndarray, enabled: bool) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(matrix.shape)).encode("ascii"))
    for row in _progress(matrix, len(matrix), "Tensor fingerprint", enabled):
        digest.update(np.ascontiguousarray(row, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _sample_negatives(
    num_items: int,
    blocked: set[int],
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    available = num_items - len([x for x in blocked if 1 <= x <= num_items])
    if available <= 0:
        raise ValueError("No candidate negatives remain after blocking history/target")

    replace = available < count
    if replace:
        pool = np.asarray(
            [item for item in range(1, num_items + 1) if item not in blocked],
            dtype=np.int64,
        )
        return rng.choice(pool, size=count, replace=True).astype(int).tolist()

    result: set[int] = set()
    while len(result) < count:
        draw = rng.integers(1, num_items + 1, size=max(32, count * 2))
        for value in draw.tolist():
            item = int(value)
            if item not in blocked:
                result.add(item)
                if len(result) == count:
                    break
    return list(result)


def _load_observed_ties(summary_path: str | None) -> dict[str, float] | None:
    if not summary_path:
        return None
    with Path(summary_path).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    best_epoch = int(summary["best_epoch"])
    row = next(
        entry for entry in summary["history"] if int(entry["epoch"]) == best_epoch
    )
    return {
        "best_epoch": best_epoch,
        "tie_case_ratio": float(row["val_tie_case_ratio"]),
        "avg_tie_items": float(row["val_avg_tie_items"]),
        "all_equal_ratio": float(row["val_all_equal_ratio"]),
    }


def main() -> None:
    args = parse_args()
    progress_enabled = args.progress == "yes" or (
        args.progress == "auto" and sys.stdout.isatty()
    )

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    data_root = str(cfg.get("data_root", "./data"))
    source = str(cfg["source"])
    data_cfg = cfg["data"]
    embedding_dir = str(data_cfg.get("embedding_dir", "semantic_embeddings"))
    max_len = int(data_cfg.get("max_len", 50))
    min_len = int(data_cfg.get("min_len", 3))
    eval_negatives = int(data_cfg.get("eval_negatives", 100))
    seed = int(data_cfg.get("seed", 2026))
    fixed_offset = int(
        cfg.get("evaluation", {}).get("fixed_negative_seed_offset", 10000)
    )

    print(f"[Audit] source={source}")
    print("[Audit] Loading the same item map and semantic tensor used by training ...")
    interactions, _, item_map = load_interactions_compat(data_root, source)

    # strict=True is intentional. If training-time mapping has missing, zero, or
    # non-finite rows, the audit should stop rather than silently repair them.
    item_features = load_semantic_embeddings(
        data_root=data_root,
        domain=source,
        item_map=item_map,
        embedding_dir=embedding_dir,
        strict=True,
    )
    matrix = item_features.detach().cpu().numpy()[1:].astype(np.float32, copy=False)
    num_items, embedding_dim = matrix.shape

    finite_rows = np.isfinite(matrix).all(axis=1)
    norms = np.linalg.norm(matrix, axis=1)
    zero_rows = norms == 0.0
    near_zero_rows = norms <= 1e-8

    if not finite_rows.all():
        raise FloatingPointError("The aligned training tensor contains NaN/Inf")
    if zero_rows.any():
        raise ValueError("The aligned training tensor contains non-padding zero rows")

    inverse_map = {int(index): str(raw_id) for raw_id, index in item_map.items()}
    if len(inverse_map) != num_items:
        raise RuntimeError(
            f"item_map has {len(inverse_map)} entries but tensor has {num_items} items"
        )

    digest_to_indices: dict[str, list[int]] = defaultdict(list)
    item_digests: list[str | None] = [None] * (num_items + 1)
    for zero_based_index, row in enumerate(
        _progress(matrix, num_items, "Hash embedding rows", progress_enabled)
    ):
        item_index = zero_based_index + 1
        digest = _row_digest(row)
        item_digests[item_index] = digest
        digest_to_indices[digest].append(item_index)

    duplicate_groups = [
        indices for indices in digest_to_indices.values() if len(indices) > 1
    ]
    duplicate_groups.sort(key=len, reverse=True)
    duplicate_items = {index for group in duplicate_groups for index in group}

    interaction_item_ids = interactions["ItemId"].astype(int).to_numpy()
    interaction_count = int(interaction_item_ids.size)
    duplicate_interactions = int(
        np.isin(interaction_item_ids, np.asarray(sorted(duplicate_items))).sum()
    )

    item_frequency = Counter(interaction_item_ids.tolist())
    group_samples = []
    for group in duplicate_groups[: max(0, int(args.max_duplicate_groups))]:
        group_samples.append(
            {
                "group_size": len(group),
                "item_indices": group[:50],
                "raw_item_ids": [inverse_map[index] for index in group[:50]],
                "interaction_count": int(
                    sum(item_frequency.get(index, 0) for index in group)
                ),
            }
        )

    tensor_sha256 = _tensor_fingerprint(matrix, progress_enabled)
    item_map_sha256 = _hash_item_map(item_map)
    protocol_digest = hashlib.sha256(
        (tensor_sha256 + ":" + item_map_sha256).encode("ascii")
    ).hexdigest()

    report: dict[str, object] = {
        "config": str(config_path),
        "source": source,
        "num_items": num_items,
        "embedding_dim": embedding_dim,
        "strict_training_tensor_checks": {
            "nonfinite_rows": int((~finite_rows).sum()),
            "zero_rows_excluding_padding": int(zero_rows.sum()),
            "near_zero_rows_excluding_padding": int(near_zero_rows.sum()),
        },
        "norms": {
            "min": float(norms.min()),
            "p01": float(np.quantile(norms, 0.01)),
            "median": float(np.median(norms)),
            "p99": float(np.quantile(norms, 0.99)),
            "max": float(norms.max()),
        },
        "exact_duplicate_vectors": {
            "duplicate_group_count": len(duplicate_groups),
            "items_in_duplicate_groups": len(duplicate_items),
            "item_ratio": (
                float(len(duplicate_items) / num_items) if num_items else 0.0
            ),
            "extra_duplicate_rows": int(
                sum(len(group) - 1 for group in duplicate_groups)
            ),
            "largest_group_size": (
                int(len(duplicate_groups[0])) if duplicate_groups else 1
            ),
            "interaction_count_on_duplicate_items": duplicate_interactions,
            "interaction_ratio_on_duplicate_items": (
                float(duplicate_interactions / interaction_count)
                if interaction_count
                else 0.0
            ),
            "largest_groups": group_samples,
        },
        "fingerprints": {
            "aligned_tensor_sha256": tensor_sha256,
            "item_map_sha256": item_map_sha256,
            "source_representation_protocol_sha256": protocol_digest,
        },
        "observed_training_summary_ties": _load_observed_ties(args.summary),
    }

    if args.simulate_validation_ties:
        print("[Audit] Reproducing fixed sampled-negative validation ties ...")
        sequences = group_user_sequences_compat(interactions, min_len=min_len)
        _, val_samples = build_source_splits_compat(
            sequences=sequences,
            max_len=max_len,
            min_len=min_len,
        )

        rng = np.random.default_rng(seed + fixed_offset)
        tie_cases = 0
        tie_items = 0
        all_equal_by_embedding = 0

        for sample in _progress(
            val_samples,
            len(val_samples),
            "Simulate validation ties",
            progress_enabled,
        ):
            history = [int(x) for x in sample["history"]]
            target = int(sample["target"])
            blocked = {item for item in history if item > 0}
            blocked.add(target)
            negatives = _sample_negatives(
                num_items=num_items,
                blocked=blocked,
                count=eval_negatives,
                rng=rng,
            )
            row = negatives + [target]
            rng.shuffle(row)  # Preserve evaluator RNG consumption.

            target_digest = item_digests[target]
            equal_count = sum(
                1
                for candidate in negatives
                if item_digests[candidate] == target_digest
            )
            if equal_count > 0:
                tie_cases += 1
                tie_items += equal_count

            if all(item_digests[candidate] == target_digest for candidate in row):
                all_equal_by_embedding += 1

        total = len(val_samples)
        predicted = {
            "num_validation_examples": total,
            "eval_negatives": eval_negatives,
            "negative_seed": seed + fixed_offset,
            "predicted_tie_case_ratio_from_exact_duplicates": (
                float(tie_cases / total) if total else 0.0
            ),
            "predicted_avg_tie_items_from_exact_duplicates": (
                float(tie_items / total) if total else 0.0
            ),
            "predicted_all_equal_ratio_from_exact_duplicates": (
                float(all_equal_by_embedding / total) if total else 0.0
            ),
        }

        observed = report["observed_training_summary_ties"]
        if observed:
            predicted["tie_case_ratio_gap_vs_observed"] = float(
                observed["tie_case_ratio"]
                - predicted["predicted_tie_case_ratio_from_exact_duplicates"]
            )
            predicted["avg_tie_items_gap_vs_observed"] = float(
                observed["avg_tie_items"]
                - predicted["predicted_avg_tie_items_from_exact_duplicates"]
            )
        report["validation_tie_simulation"] = predicted

    output = (
        Path(args.output)
        if args.output
        else Path("results")
        / "data_quality"
        / f"{source}_semantic_embedding_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[Audit] Saved: {output}")


if __name__ == "__main__":
    main()
