from __future__ import annotations

from collections.abc import Iterable
import os

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - training still works without tqdm
    tqdm = None



_PROGRESS_TRUE = {"1", "true", "yes", "on", "show", "enabled"}
_PROGRESS_FALSE = {"0", "false", "no", "off", "hide", "disabled", "quiet", "batch"}


def _resolve_progress_enabled(explicit: bool | None = None) -> bool:
    """
    Resolve progress display.

    Priority:
      1. ISDDG_PROGRESS environment variable;
      2. explicit argument;
      3. enabled by default for a directly launched model.

    Batch launchers should temporarily set ISDDG_PROGRESS=0.
    """
    raw = os.getenv("ISDDG_PROGRESS")
    if raw is not None:
        normalized = raw.strip().lower()
        if normalized in _PROGRESS_TRUE:
            return True
        if normalized in _PROGRESS_FALSE:
            return False
        raise ValueError(
            "ISDDG_PROGRESS must be one of "
            f"{sorted(_PROGRESS_TRUE | _PROGRESS_FALSE)}, got {raw!r}"
        )
    if explicit is not None:
        return bool(explicit)
    return True


def _safe_total(loader: Iterable, max_batches: int) -> int | None:
    try:
        total = len(loader)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return max_batches if max_batches > 0 else None
    if max_batches > 0:
        total = min(int(total), int(max_batches))
    return int(total)

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


def _rank_with_ties(scores: np.ndarray, target_index: int, policy: str) -> int:
    target = float(scores[target_index])
    greater = int(np.sum(scores > target))
    equal_others = int(np.sum(scores == target)) - 1

    if policy == "worst":
        return greater + max(equal_others, 0)
    if policy == "best":
        return greater
    if policy == "average":
        return int(round(greater + max(equal_others, 0) / 2.0))
    raise ValueError(f"Unknown tie policy: {policy}")


@torch.no_grad()
def evaluate_bert4rec_family(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    num_items: int,
    ranking: str = "sampled",
    eval_negatives: int = 100,
    ks: tuple[int, ...] = (10, 20),
    tie_policy: str = "worst",
    seed: int = 12026,
    max_batches: int = 0,
    show_progress: bool | None = None,
    progress_desc: str = "Evaluate",
) -> dict[str, float]:
    """
    Deterministic source/target evaluator.

    The same seed can be reused at every epoch, so validation candidates do not
    change while early stopping compares checkpoints.
    """
    model.eval()
    rng = np.random.default_rng(int(seed))
    ranks: list[int] = []
    all_equal_count = 0
    tie_case_count = 0
    tie_items_total = 0

    progress_enabled = _resolve_progress_enabled(show_progress)
    progress_bar = None
    iterator = loader
    if progress_enabled and tqdm is not None:
        progress_bar = tqdm(
            loader,
            total=_safe_total(loader, max_batches),
            desc=progress_desc,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
        )
        iterator = progress_bar

    try:
        for batch_index, batch in enumerate(iterator):
            if max_batches > 0 and batch_index >= max_batches:
                break

            histories = batch["history"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            history_lists = histories.detach().cpu().tolist()
            target_list = targets.detach().cpu().tolist()

            if ranking == "full":
                score_tensor = model.score_all_items(histories)
                scores = score_tensor.detach().float().cpu().numpy()
                if not np.isfinite(scores).all():
                    raise FloatingPointError(
                        f"Non-finite validation scores in batch {batch_index}"
                    )

                for row, target in enumerate(target_list):
                    row_scores = scores[row].copy()
                    blocked = {int(x) for x in history_lists[row] if int(x) > 0}
                    for item in blocked:
                        if 1 <= item <= num_items and item != int(target):
                            row_scores[item - 1] = -np.inf
                    target_index = int(target) - 1
                    finite = np.isfinite(row_scores)
                    if not finite[target_index]:
                        raise FloatingPointError("Target score is non-finite")
                    valid_scores = row_scores[finite]
                    target_score = row_scores[target_index]
                    greater = int(np.sum(valid_scores > target_score))
                    equal_others = int(np.sum(valid_scores == target_score)) - 1
                    if equal_others > 0:
                        tie_case_count += 1
                        tie_items_total += equal_others
                    if valid_scores.size > 0 and np.all(valid_scores == valid_scores[0]):
                        all_equal_count += 1
                    if tie_policy == "worst":
                        rank = greater + max(equal_others, 0)
                    elif tie_policy == "best":
                        rank = greater
                    else:
                        rank = int(round(greater + max(equal_others, 0) / 2.0))
                    ranks.append(rank)
                continue

            if ranking != "sampled":
                raise ValueError("ranking must be 'sampled' or 'full'")

            candidates: list[list[int]] = []
            target_indices: list[int] = []
            for history, target in zip(history_lists, target_list):
                blocked = {int(x) for x in history if int(x) > 0}
                blocked.add(int(target))
                negatives = _sample_negatives(
                    num_items=num_items,
                    blocked=blocked,
                    count=int(eval_negatives),
                    rng=rng,
                )
                row = negatives + [int(target)]
                rng.shuffle(row)
                candidates.append(row)
                target_indices.append(row.index(int(target)))

            candidate_tensor = torch.as_tensor(
                candidates, dtype=torch.long, device=device
            )
            score_tensor = model.score_candidates(histories, candidate_tensor)
            scores = score_tensor.detach().float().cpu().numpy()
            if not np.isfinite(scores).all():
                raise FloatingPointError(
                    f"Non-finite validation scores in batch {batch_index}; "
                    "the checkpoint is invalid and must not be saved"
                )

            for row_scores, target_index in zip(scores, target_indices):
                equal_others = int(np.sum(row_scores == row_scores[target_index])) - 1
                if equal_others > 0:
                    tie_case_count += 1
                    tie_items_total += equal_others
                if np.all(row_scores == row_scores[0]):
                    all_equal_count += 1
                ranks.append(_rank_with_ties(row_scores, target_index, tie_policy))
            if progress_bar is not None:
                progress_bar.set_postfix(
                    examples=len(ranks),
                    ties=tie_case_count,
                    refresh=False,
                )

    finally:
        if progress_bar is not None:
            progress_bar.close()

    if not ranks:
        raise RuntimeError("Evaluation produced no examples")

    rank_array = np.asarray(ranks, dtype=np.int64)
    metrics: dict[str, float] = {"num_examples": float(len(ranks))}
    for k in sorted(set(int(x) for x in ks)):
        hit = rank_array < k
        metrics[f"Recall@{k}"] = float(hit.mean())
        metrics[f"NDCG@{k}"] = float(
            np.where(hit, 1.0 / np.log2(rank_array + 2.0), 0.0).mean()
        )
        metrics[f"MRR@{k}"] = float(
            np.where(hit, 1.0 / (rank_array + 1.0), 0.0).mean()
        )

    metrics["all_equal_ratio"] = float(all_equal_count / len(ranks))
    metrics["tie_case_ratio"] = float(tie_case_count / len(ranks))
    metrics["avg_tie_items"] = float(tie_items_total / len(ranks))
    metrics["mean_rank_1based"] = float((rank_array + 1).mean())
    return metrics
