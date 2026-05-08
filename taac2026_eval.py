from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from taac2026_data import TAAC2026Batcher, move_batch_to_device
from taac2026_infer import _load_model, find_checkpoint, write_outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local labeled TAAC2026 evaluation with feedback logging")
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--model_output_path", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--feedback_dir", type=str, default="feedbacks/records")
    parser.add_argument("--experiment_name", type=str, default="taac2026-tail-eval")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--output_name", type=str, default="labeled_result.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _label_array(batch: dict, size: int) -> np.ndarray:
    if "labels" not in batch:
        raise ValueError("Evaluation data must include a supported label column.")
    labels = batch["labels"].detach().cpu().numpy().astype(np.float32)
    if labels.shape[0] != size:
        raise ValueError(f"Label count mismatch: labels={labels.shape[0]} predictions={size}")
    return labels


def score_stats(scores: np.ndarray) -> dict[str, float | int | None]:
    finite = scores[np.isfinite(scores)]
    stats: dict[str, float | int | None] = {
        "count": int(scores.size),
        "nan_count": int(np.isnan(scores).sum()),
        "inf_count": int(np.isinf(scores).sum()),
    }
    if finite.size == 0:
        stats.update({"min": None, "max": None, "mean": None, "std": None, "p01": None, "p50": None, "p99": None})
        return stats
    stats.update(
        {
            "min": round(float(np.min(finite)), 8),
            "max": round(float(np.max(finite)), 8),
            "mean": round(float(np.mean(finite)), 8),
            "std": round(float(np.std(finite)), 8),
            "p01": round(float(np.quantile(finite, 0.01)), 8),
            "p50": round(float(np.quantile(finite, 0.50)), 8),
            "p99": round(float(np.quantile(finite, 0.99)), 8),
        }
    )
    return stats


def choose_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return 0.5
    if np.unique(finite_scores).size <= 512:
        candidates = np.unique(finite_scores)
    else:
        candidates = np.quantile(finite_scores, np.linspace(0.01, 0.99, 199))
    best_threshold = 0.5
    best_acc = -1.0
    for threshold in candidates:
        preds = scores >= threshold
        acc = float(np.mean(preds == labels))
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
    return min(max(best_threshold, 0.0), 1.0)


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    if labels.size == 0:
        raise ValueError("No labeled examples were evaluated.")
    if labels.shape[0] != scores.shape[0]:
        raise ValueError(f"Metric count mismatch: labels={labels.shape[0]} scores={scores.shape[0]}")
    if np.any(~np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("Scores must be finite and non-negative.")

    labels = (labels > 0.5).astype(np.int64)
    threshold_used = choose_accuracy_threshold(labels, scores) if threshold < 0 else float(threshold)
    threshold_used = min(max(threshold_used, 0.0), 1.0)
    preds = (scores >= threshold_used).astype(np.int64)

    out: dict[str, float | int | None] = {
        "num_examples": int(labels.size),
        "positive_rate": round(float(np.mean(labels)), 8),
        "threshold": round(threshold_used, 8),
        "accuracy": round(float(accuracy_score(labels, preds)), 8),
    }
    out["auc"] = round(float(roc_auc_score(labels, scores)), 8) if np.unique(labels).size > 1 else None
    out["logloss"] = round(float(log_loss(labels, np.clip(scores, 1e-7, 1.0 - 1e-7), labels=[0, 1])), 8)
    out["score_stats"] = score_stats(scores)
    return out


def write_feedback_record(
    feedback_dir: Path,
    experiment_name: str,
    summary: dict,
    metrics: dict,
    config: dict,
) -> Path:
    feedback_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H%M")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in experiment_name.lower()).strip("-")
    path = feedback_dir / f"{now}_{safe_name or 'taac2026-eval'}.md"
    content = f"""# TAAC2026 Local Evaluation

## Date

{datetime.now().isoformat(timespec="seconds")}

## Experiment

{experiment_name}

## Configuration

```json
{json.dumps(config, indent=2, ensure_ascii=False)}
```

## Result

```json
{json.dumps(metrics, indent=2, ensure_ascii=False)}
```

## Log Notes

- Checkpoint: {summary["checkpoint"]}
- Evaluated examples: {metrics["num_examples"]}
- Elapsed seconds: {summary["elapsed_sec"]}
- Result JSON: {summary["result_json"]}
- Metrics JSON: {summary["metrics_json"]}

## Decision

Use this local result as the gate before any platform evaluation slot.

## Next Action

Compare AUC, accuracy, logloss, and score distribution against the current best checkpoint.
"""
    path.write_text(content, encoding="utf-8")
    return path


def evaluate(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    eval_data_path = args.eval_data_path or os.environ.get("EVAL_DATA_PATH")
    result_dir = Path(args.result_dir or os.environ.get("EVAL_RESULT_PATH", "./eval_result"))
    ckpt_root = args.model_output_path or os.environ.get("MODEL_OUTPUT_PATH")
    ckpt_path = find_checkpoint(ckpt_root)
    device = torch.device(args.device)

    model, config, saved_args = _load_model(ckpt_path, device)
    batcher = TAAC2026Batcher(
        eval_data_path,
        config,
        hash_size=int(saved_args.get("hash_size", 200_000)),
        batch_size=args.batch_size,
        include_label=True,
        seq_len=int(saved_args.get("seq_len", 32)),
    )

    scores: list[float] = []
    labels: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    start = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(batcher):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batch = move_batch_to_device(batch, device)
            logits = model(
                batch["ns_tokens"],
                batch["ns_dense"],
                batch["seq_tokens"],
                batch["seq_mask"],
                batch["time_features"],
            )
            batch_scores = torch.sigmoid(logits).detach().cpu().numpy().astype(float)
            scores.extend(batch_scores.tolist())
            labels.append(_label_array(batch, batch_scores.shape[0]))
            user_ids = batch.get("user_id", [""] * len(batch_scores))
            item_ids = batch.get("item_id", [""] * len(batch_scores))
            rows.extend({"user_id": u, "item_id": i} for u, i in zip(user_ids, item_ids))

    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = np.concatenate(labels) if labels else np.asarray([], dtype=np.float32)
    metrics = compute_metrics(label_arr, score_arr, args.threshold)
    write_outputs(result_dir, rows, scores, args.output_name)

    summary = {
        "checkpoint": str(ckpt_path),
        "eval_data_path": str(eval_data_path),
        "elapsed_sec": round(time.time() - start, 3),
        "result_json": str(result_dir / args.output_name),
        "metrics_json": str(result_dir / "labeled_eval_summary.json"),
        "feedback_record": None,
        "metrics": metrics,
    }
    config_summary = {
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "max_batches": args.max_batches,
        "saved_train_args": saved_args,
    }
    summary["feedback_record"] = str(
        write_feedback_record(Path(args.feedback_dir), args.experiment_name, summary, metrics, config_summary)
    )
    (result_dir / "labeled_eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    evaluate()
