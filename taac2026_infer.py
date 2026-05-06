from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from taac2026_data import TAAC2026Batcher, move_batch_to_device
from taac2026_model import TAAC2026Scorer
from taac2026_schema import TAACFeatureConfig


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAAC2026 pCVR inference")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_name", type=str, default="result.json")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unsupported TAAC2026 inference args: {unknown}")
    return args


def find_checkpoint(model_dir: str | Path | None) -> Path:
    if model_dir is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    root = Path(model_dir)
    candidates = sorted(root.glob("taac2026*.pt"))
    if not candidates:
        candidates = sorted(root.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {root}")
    return candidates[-1]


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[TAAC2026Scorer, TAACFeatureConfig, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    config = TAACFeatureConfig.from_dict(ckpt["feature_config"])
    saved_args = ckpt.get("args", {})
    model = TAAC2026Scorer(
        num_cat_fields=len(config.cat_columns),
        num_dense_fields=len(config.dense_columns),
        num_token_fields=len(config.token_fields),
        hash_size=int(saved_args.get("hash_size", 200_000)),
        dense_hash_size=int(saved_args.get("dense_hash_size", 4_096)),
        hidden_dim=int(saved_args.get("hidden_units", 128)),
        num_layers=int(saved_args.get("num_blocks", 2)),
        num_heads=int(saved_args.get("num_heads", 4)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, saved_args


def write_outputs(
    result_dir: Path,
    rows: list[dict[str, object]],
    scores: list[float],
    output_name: str,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row, score in zip(rows, scores):
        item = {"score": float(score)}
        if "user_id" in row:
            item["user_id"] = row["user_id"]
        if "item_id" in row:
            item["item_id"] = row["item_id"]
        records.append(item)

    payload = {"predictions": records}
    (result_dir / output_name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with (result_dir / "submission.csv").open("w", encoding="utf-8") as f:
        f.write("user_id,item_id,score\n")
        for row, score in zip(rows, scores):
            f.write(f"{row.get('user_id', '')},{row.get('item_id', '')},{float(score):.8f}\n")

    np.save(result_dir / "scores.npy", np.asarray(scores, dtype=np.float32))


def infer(argv: list[str] | None = None) -> dict:
    args = get_args(argv)
    eval_data_path = os.environ.get("EVAL_DATA_PATH")
    result_dir = Path(os.environ.get("EVAL_RESULT_PATH", "./eval_result"))
    ckpt_path = find_checkpoint(os.environ.get("MODEL_OUTPUT_PATH"))
    device = torch.device(args.device)

    model, config, saved_args = _load_model(ckpt_path, device)
    hash_size = int(saved_args.get("hash_size", 200_000))

    batcher = TAAC2026Batcher(
        eval_data_path,
        config,
        hash_size=hash_size,
        batch_size=args.batch_size,
        include_label=False,
    )

    scores: list[float] = []
    rows: list[dict[str, object]] = []
    start = time.time()
    with torch.no_grad():
        for batch in batcher:
            batch = move_batch_to_device(batch, device)
            logits = model(
                batch["cat_tokens"],
                batch["dense_values"],
                batch["dense_mask"],
                batch["time_features"],
            )
            batch_scores = torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist()
            user_ids = batch.get("user_id", [""] * len(batch_scores))
            item_ids = batch.get("item_id", [""] * len(batch_scores))
            scores.extend(batch_scores)
            rows.extend({"user_id": u, "item_id": i} for u, i in zip(user_ids, item_ids))

    write_outputs(result_dir, rows, scores, args.output_name)
    summary = {
        "num_predictions": len(scores),
        "checkpoint": str(ckpt_path),
        "elapsed_sec": round(time.time() - start, 3),
        "result_json": str(result_dir / args.output_name),
        "submission_csv": str(result_dir / "submission.csv"),
    }
    (result_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    infer()
