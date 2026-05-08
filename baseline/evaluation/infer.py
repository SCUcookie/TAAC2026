from __future__ import annotations

import argparse
import json
import logging
import os
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_TRAINING_DIR = _THIS_DIR.parent / "training"
if str(_TRAINING_DIR) not in sys.path:
    sys.path.append(str(_TRAINING_DIR))

try:
    from .dataset import FeatureSchema, NUM_TIME_BUCKETS, PCVRParquetDataset
    from .model import ModelInput, PCVRHyFormer
    from .utils import create_logger
except ImportError:
    from dataset import FeatureSchema, NUM_TIME_BUCKETS, PCVRParquetDataset
    from model import ModelInput, PCVRHyFormer
    from utils import create_logger


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for _, offset, length in schema.entries:
        vocab_size = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vocab_size, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer inference")
    parser.add_argument("--eval_data_dir", type=str, default=None,
                        help="Evaluation data directory or parquet file (env: EVAL_DATA_PATH)")
    parser.add_argument("--schema_path", type=str, default=None,
                        help="Schema JSON path (default: <ckpt_dir>/schema.json, else <eval_data_dir>/schema.json)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Checkpoint root directory (env: MODEL_OUTPUT_PATH or TRAIN_CKPT_PATH)")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="Output directory (env: EVAL_RESULT_PATH)")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_name", type=str, default="predictions.json")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unsupported inference args: {unknown}")
    args.eval_data_dir = os.environ.get("EVAL_DATA_PATH", args.eval_data_dir)
    args.model_dir = os.environ.get("MODEL_OUTPUT_PATH", os.environ.get("TRAIN_CKPT_PATH", args.model_dir))
    args.result_dir = os.environ.get("EVAL_RESULT_PATH", args.result_dir)
    return args


def find_checkpoint_dir(model_dir: str | None) -> Path:
    if not model_dir:
        raise ValueError("MODEL_OUTPUT_PATH or TRAIN_CKPT_PATH is not set")
    root = Path(model_dir)
    if root.is_file() and root.name == "model.pt":
        return root.parent
    best_dirs = sorted(p for p in root.glob("global_step*.best_model") if (p / "model.pt").exists())
    if best_dirs:
        return best_dirs[-1]
    nested = sorted(p.parent for p in root.rglob("model.pt"))
    if nested:
        return nested[-1]
    raise FileNotFoundError(f"No model.pt found under {root}")


def load_train_config(ckpt_dir: Path) -> Dict[str, Any]:
    cfg_path = ckpt_dir / "train_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing train_config.json in {ckpt_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_schema_path(args: argparse.Namespace, ckpt_dir: Path) -> Path:
    if args.schema_path:
        return Path(args.schema_path)
    ckpt_schema = ckpt_dir / "schema.json"
    if ckpt_schema.exists():
        return ckpt_schema
    if not args.eval_data_dir:
        raise ValueError("EVAL_DATA_PATH is required when schema.json is not bundled with the checkpoint")
    return Path(args.eval_data_dir) / "schema.json"


def parse_seq_max_lens(raw: Any) -> Dict[str, int]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    seq_max_lens: Dict[str, int] = {}
    for pair in str(raw).split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, value = pair.split(":")
        seq_max_lens[key.strip()] = int(value.strip())
    return seq_max_lens


def build_ns_groups(pcvr_dataset: PCVRParquetDataset, cfg: Dict[str, Any], ckpt_dir: Path) -> Tuple[List[List[int]], List[List[int]]]:
    ns_groups_path = cfg.get("ns_groups_json")
    if ns_groups_path:
        candidate = Path(ns_groups_path)
        if not candidate.is_absolute():
            candidate = ckpt_dir / candidate
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                ns_groups_cfg = json.load(f)
            user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
            item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
            user_ns_groups = [[user_fid_to_idx[fid] for fid in fids] for fids in ns_groups_cfg["user_ns_groups"].values()]
            item_ns_groups = [[item_fid_to_idx[fid] for fid in fids] for fids in ns_groups_cfg["item_ns_groups"].values()]
            return user_ns_groups, item_ns_groups
    user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
    item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]
    return user_ns_groups, item_ns_groups


def make_model_input(device_batch: Dict[str, Any], device: torch.device) -> ModelInput:
    seq_domains = device_batch["_seq_domains"]
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f"{domain}_len"]
        batch_size = device_batch[domain].shape[0]
        seq_len = device_batch[domain].shape[2]
        seq_time_buckets[domain] = device_batch.get(
            f"{domain}_time_bucket",
            torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        )
    return ModelInput(
        user_int_feats=device_batch["user_int_feats"],
        item_int_feats=device_batch["item_int_feats"],
        user_dense_feats=device_batch["user_dense_feats"],
        item_dense_feats=device_batch["item_dense_feats"],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
    )


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def write_outputs(
    result_dir: Path,
    user_ids: List[str],
    scores: List[float],
    output_name: str,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    predictions = {
        "predictions": {
            str(user_id): float(score)
            for user_id, score in zip(user_ids, scores)
        }
    }

    (result_dir / output_name).write_text(
        json.dumps(predictions, ensure_ascii=False),
        encoding="utf-8",
    )

    with (result_dir / "submission.csv").open("w", encoding="utf-8") as f:
        f.write("user_id,score\n")
        for user_id, score in zip(user_ids, scores):
            f.write(f"{user_id},{float(score):.8f}\n")

    np.save(result_dir / "scores.npy", np.asarray(scores, dtype=np.float32))


def score_stats(scores: List[float]) -> Dict[str, Any]:
    if not scores:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "nan_count": 0,
            "inf_count": 0,
        }

    arr = np.asarray(scores, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    finite = arr[finite_mask]
    stats: Dict[str, Any] = {
        "count": int(arr.size),
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
    }
    if finite.size == 0:
        stats.update({
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p99": None,
        })
        return stats

    stats.update({
        "min": round(float(np.min(finite)), 8),
        "max": round(float(np.max(finite)), 8),
        "mean": round(float(np.mean(finite)), 8),
        "std": round(float(np.std(finite)), 8),
        "p01": round(float(np.quantile(finite, 0.01)), 8),
        "p50": round(float(np.quantile(finite, 0.50)), 8),
        "p99": round(float(np.quantile(finite, 0.99)), 8),
    })
    return stats


def print_platform_event(name: str, payload: Dict[str, Any]) -> None:
    print(f"PLATFORM_{name} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)


def compact_file_status(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def infer() -> Dict[str, Any]:
    args = parse_args()
    print_platform_event("INFER_START", {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": os.getcwd(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    })
    if not args.eval_data_dir:
        raise ValueError("EVAL_DATA_PATH or --eval_data_dir is required")

    result_dir = Path(args.result_dir or "./eval_result")
    ckpt_dir = find_checkpoint_dir(args.model_dir)
    schema_path = resolve_schema_path(args, ckpt_dir)
    train_cfg = load_train_config(ckpt_dir)
    device = torch.device(args.device)
    print_platform_event("CHECKPOINT_STATUS", {
        "checkpoint_dir": str(ckpt_dir),
        "model_file": compact_file_status(ckpt_dir / "model.pt"),
        "schema_file": compact_file_status(schema_path),
        "train_config_file": compact_file_status(ckpt_dir / "train_config.json"),
    })

    result_dir.mkdir(parents=True, exist_ok=True)
    create_logger(str(result_dir / "infer.log"))
    logging.info("Using checkpoint directory: %s", ckpt_dir)
    logging.info("Using schema path: %s", schema_path)

    seq_max_lens = parse_seq_max_lens(train_cfg.get("seq_max_lens"))
    batch_size = int(train_cfg.get("batch_size", args.batch_size) or args.batch_size)
    train_cfg_summary = {
        "seq_max_lens": train_cfg.get("seq_max_lens"),
        "d_model": train_cfg.get("d_model"),
        "num_hyformer_blocks": train_cfg.get("num_hyformer_blocks"),
        "num_heads": train_cfg.get("num_heads"),
        "num_queries": train_cfg.get("num_queries"),
        "seq_encoder_type": train_cfg.get("seq_encoder_type"),
        "loss_type": train_cfg.get("loss_type"),
        "batch_size": batch_size,
        "emb_skip_threshold": train_cfg.get("emb_skip_threshold"),
    }
    print_platform_event("INFER_CONFIG", {
        "checkpoint_dir": str(ckpt_dir),
        "model_path": str(ckpt_dir / "model.pt"),
        "schema_path": str(schema_path),
        "eval_data_dir": str(args.eval_data_dir),
        "result_dir": str(result_dir),
        "device": str(device),
        "num_workers": args.num_workers,
        "train_config": train_cfg_summary,
    })

    dataset = PCVRParquetDataset(
        parquet_path=args.eval_data_dir,
        schema_path=str(schema_path),
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        clip_vocab=True,
        is_training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print_platform_event("DATASET_CONFIG", {
        "batch_size": batch_size,
        "num_rows_estimate": getattr(dataset, "num_rows", None),
        "num_row_groups": len(getattr(dataset, "_rg_list", [])),
        "seq_domains": dataset.seq_domains,
        "seq_max_lens": seq_max_lens,
        "user_int_features": len(dataset.user_int_schema.entries),
        "item_int_features": len(dataset.item_int_schema.entries),
        "user_dense_dim": dataset.user_dense_schema.total_dim,
        "item_dense_dim": dataset.item_dense_schema.total_dim,
    })

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
    print_platform_event("MODEL_READY", {
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "num_ns": getattr(model, "num_ns", None),
        "device": str(device),
    })

    all_user_ids: List[str] = []
    scores: List[float] = []
    start = time.time()
    first_batch_event = False
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            model_input = make_model_input(batch, device)
            logits, _ = model.predict(model_input)
            batch_scores = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy().astype(float).tolist()
            user_ids = batch.get("user_id", [""] * len(batch_scores))
            scores.extend(batch_scores)
            all_user_ids.extend(str(user_id) for user_id in user_ids)
            if not first_batch_event:
                print_platform_event("FIRST_BATCH", {
                    "batch_predictions": len(batch_scores),
                    "sample_user_ids": [str(x) for x in user_ids[:5]],
                    "sample_scores": [round(float(x), 8) for x in batch_scores[:5]],
                })
                first_batch_event = True

    if len(all_user_ids) != len(scores):
        raise RuntimeError(
            f"Prediction count mismatch: {len(all_user_ids)} user_ids vs {len(scores)} scores"
        )
    if len(set(all_user_ids)) != len(all_user_ids):
        raise RuntimeError("Duplicate user_id values detected in evaluation output")
    if any((not math.isfinite(float(score))) or float(score) < 0.0 for score in scores):
        raise RuntimeError("Invalid prediction score detected: scores must be finite and non-negative")

    write_outputs(result_dir, all_user_ids, scores, args.output_name)
    stats = score_stats(scores)
    result_json = result_dir / args.output_name
    submission_csv = result_dir / "submission.csv"
    scores_npy = result_dir / "scores.npy"
    summary = {
        "num_predictions": len(scores),
        "num_unique_user_ids": len(set(all_user_ids)),
        "checkpoint_dir": str(ckpt_dir),
        "elapsed_sec": round(time.time() - start, 3),
        "output_shape": "predictions_dict_user_id_to_float",
        "result_json": str(result_json),
        "submission_csv": str(submission_csv),
        "scores_npy": str(scores_npy),
        "result_files": {
            "json": compact_file_status(result_json),
            "csv": compact_file_status(submission_csv),
            "scores": compact_file_status(scores_npy),
        },
        "score_stats": stats,
    }
    (result_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_platform_event("INFER_SUMMARY", summary)
    return summary


if __name__ == "__main__":
    infer()
