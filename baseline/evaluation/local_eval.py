from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_THIS_DIR = Path(__file__).resolve().parent
_TRAINING_DIR = _THIS_DIR.parent / "training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.append(str(_TRAINING_DIR))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from torch.utils.data import DataLoader

try:
    from .dataset import NUM_TIME_BUCKETS, PCVRParquetDataset
    from .infer import (
        build_feature_specs,
        build_ns_groups,
        find_checkpoint_dir,
        load_train_config,
        make_model_input,
        move_batch_to_device,
        parse_seq_max_lens,
        resolve_schema_path,
        write_outputs,
    )
    from .model import PCVRHyFormer
except ImportError:
    from dataset import NUM_TIME_BUCKETS, PCVRParquetDataset
    from infer import (
        build_feature_specs,
        build_ns_groups,
        find_checkpoint_dir,
        load_train_config,
        make_model_input,
        move_batch_to_device,
        parse_seq_max_lens,
        resolve_schema_path,
        write_outputs,
    )
    from model import PCVRHyFormer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline labeled local evaluation with feedback logging")
    parser.add_argument("--eval_data_dir", type=str, default=None)
    parser.add_argument("--schema_path", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--feedback_dir", type=str, default="feedbacks/records")
    parser.add_argument("--experiment_name", type=str, default="baseline-local-eval")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--output_name", type=str, default="labeled_predictions.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unsupported local-eval args: {unknown}")
    args.eval_data_dir = os.environ.get("EVAL_DATA_PATH", args.eval_data_dir)
    args.model_dir = os.environ.get("MODEL_OUTPUT_PATH", os.environ.get("TRAIN_CKPT_PATH", args.model_dir))
    args.result_dir = os.environ.get("EVAL_RESULT_PATH", args.result_dir)
    return args


def build_model(
    dataset: PCVRParquetDataset,
    train_cfg: Dict[str, Any],
    ckpt_dir: Path,
    device: torch.device,
) -> PCVRHyFormer:
    user_ns_groups, item_ns_groups = build_ns_groups(dataset, train_cfg, ckpt_dir)
    model_args = {
        "user_int_feature_specs": build_feature_specs(dataset.user_int_schema, dataset.user_int_vocab_sizes),
        "item_int_feature_specs": build_feature_specs(dataset.item_int_schema, dataset.item_int_vocab_sizes),
        "user_dense_dim": dataset.user_dense_schema.total_dim,
        "item_dense_dim": dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": int(train_cfg.get("d_model", 64)),
        "emb_dim": int(train_cfg.get("emb_dim", 64)),
        "num_queries": int(train_cfg.get("num_queries", 1)),
        "num_hyformer_blocks": int(train_cfg.get("num_hyformer_blocks", 2)),
        "num_heads": int(train_cfg.get("num_heads", 4)),
        "seq_encoder_type": train_cfg.get("seq_encoder_type", "transformer"),
        "hidden_mult": int(train_cfg.get("hidden_mult", 4)),
        "dropout_rate": 0.0,
        "seq_top_k": int(train_cfg.get("seq_top_k", 50)),
        "seq_causal": bool(train_cfg.get("seq_causal", False)),
        "action_num": int(train_cfg.get("action_num", 1)),
        "num_time_buckets": NUM_TIME_BUCKETS if bool(train_cfg.get("use_time_buckets", True)) else 0,
        "rank_mixer_mode": train_cfg.get("rank_mixer_mode", "full"),
        "use_rope": bool(train_cfg.get("use_rope", False)),
        "rope_base": float(train_cfg.get("rope_base", 10000.0)),
        "emb_skip_threshold": int(train_cfg.get("emb_skip_threshold", 0)),
        "seq_id_threshold": int(train_cfg.get("seq_id_threshold", 10000)),
        "ns_tokenizer_type": train_cfg.get("ns_tokenizer_type", "rankmixer"),
        "user_ns_tokens": int(train_cfg.get("user_ns_tokens", 0)),
        "item_ns_tokens": int(train_cfg.get("item_ns_tokens", 0)),
    }
    model = PCVRHyFormer(**model_args).to(device)
    state_dict = torch.load(ckpt_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def choose_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        return 0.5
    candidates = np.unique(finite_scores) if np.unique(finite_scores).size <= 512 else np.quantile(
        finite_scores, np.linspace(0.01, 0.99, 199)
    )
    best_threshold = 0.5
    best_acc = -1.0
    for threshold in candidates:
        acc = float(np.mean((scores >= threshold) == labels))
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)
    return min(max(best_threshold, 0.0), 1.0)


def score_stats(scores: np.ndarray) -> Dict[str, Any]:
    finite = scores[np.isfinite(scores)]
    stats: Dict[str, Any] = {
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


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    if labels.size == 0:
        raise ValueError("No labeled examples were evaluated.")
    labels = (labels > 0.5).astype(np.int64)
    if np.any(~np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("Scores must be finite and non-negative.")
    threshold_used = choose_accuracy_threshold(labels, scores) if threshold < 0 else float(threshold)
    threshold_used = min(max(threshold_used, 0.0), 1.0)
    preds = (scores >= threshold_used).astype(np.int64)
    return {
        "num_examples": int(labels.size),
        "positive_rate": round(float(np.mean(labels)), 8),
        "threshold": round(threshold_used, 8),
        "accuracy": round(float(accuracy_score(labels, preds)), 8),
        "auc": round(float(roc_auc_score(labels, scores)), 8) if np.unique(labels).size > 1 else None,
        "logloss": round(float(log_loss(labels, np.clip(scores, 1e-7, 1.0 - 1e-7), labels=[0, 1])), 8),
        "score_stats": score_stats(scores),
    }


def write_feedback_record(feedback_dir: Path, experiment_name: str, summary: Dict[str, Any]) -> Path:
    feedback_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in experiment_name.lower()).strip("-")
    path = feedback_dir / f"{now:%Y-%m-%d_%H%M}_{safe_name or 'baseline-local-eval'}.md"
    content = f"""# Baseline Local Evaluation

## Date

{now.isoformat(timespec="seconds")}

## Experiment

{experiment_name}

## Configuration

```json
{json.dumps(summary["config"], indent=2, ensure_ascii=False)}
```

## Result

```json
{json.dumps(summary["metrics"], indent=2, ensure_ascii=False)}
```

## Log Notes

- Benchmark to beat: 0.8133
- Checkpoint: {summary["checkpoint_dir"]}
- Result JSON: {summary["result_json"]}
- Metrics JSON: {summary["metrics_json"]}

## Decision

Use this result to decide whether the checkpoint is stronger than the 0.8133 baseline.

## Next Action

Promote only if local metrics and platform-safe output checks justify spending an evaluation slot.
"""
    path.write_text(content, encoding="utf-8")
    return path


def evaluate() -> Dict[str, Any]:
    args = parse_args()
    if not args.eval_data_dir:
        raise ValueError("EVAL_DATA_PATH or --eval_data_dir is required")

    result_dir = Path(args.result_dir or "./eval_result")
    ckpt_dir = find_checkpoint_dir(args.model_dir)
    schema_path = resolve_schema_path(args, ckpt_dir)
    train_cfg = load_train_config(ckpt_dir)
    seq_max_lens = parse_seq_max_lens(train_cfg.get("seq_max_lens"))
    batch_size = int(train_cfg.get("batch_size", args.batch_size) or args.batch_size)
    device = torch.device(args.device)

    dataset = PCVRParquetDataset(
        parquet_path=args.eval_data_dir,
        schema_path=str(schema_path),
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        clip_vocab=True,
        is_training=True,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    model = build_model(dataset, train_cfg, ckpt_dir, device)

    labels: List[np.ndarray] = []
    scores: List[float] = []
    user_ids: List[str] = []
    start = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            batch = move_batch_to_device(batch, device)
            logits, _ = model.predict(make_model_input(batch, device))
            batch_scores = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy().astype(float)
            scores.extend(batch_scores.tolist())
            labels.append(batch["label"].detach().cpu().numpy())
            user_ids.extend(str(user_id) for user_id in batch.get("user_id", [""] * len(batch_scores)))

    score_arr = np.asarray(scores, dtype=np.float64)
    label_arr = np.concatenate(labels) if labels else np.asarray([], dtype=np.int64)
    metrics = compute_metrics(label_arr, score_arr, args.threshold)
    write_outputs(result_dir, user_ids, scores, args.output_name)

    summary: Dict[str, Any] = {
        "checkpoint_dir": str(ckpt_dir),
        "eval_data_dir": str(args.eval_data_dir),
        "elapsed_sec": round(time.time() - start, 3),
        "result_json": str(result_dir / args.output_name),
        "metrics_json": str(result_dir / "local_eval_summary.json"),
        "metrics": metrics,
        "config": {
            "batch_size": batch_size,
            "threshold": args.threshold,
            "max_batches": args.max_batches,
            "seq_max_lens": seq_max_lens,
            "train_config": train_cfg,
        },
    }
    summary["feedback_record"] = str(write_feedback_record(Path(args.feedback_dir), args.experiment_name, summary))
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "local_eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    evaluate()
