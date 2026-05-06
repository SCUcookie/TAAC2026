from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from taac2026_data import TAAC2026Batcher, move_batch_to_device
from taac2026_model import TAAC2026Scorer
from taac2026_schema import build_feature_config, find_parquet_files, read_parquet_columns


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAAC2026 pCVR training")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--hash_size", type=int, default=200_000)
    parser.add_argument("--dense_hash_size", type=int, default=4_096)
    parser.add_argument("--max_train_batches", type=int, default=6000)
    parser.add_argument("--val_batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unsupported TAAC2026 training args: {unknown}")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_auc(labels: list[np.ndarray], preds: list[np.ndarray]) -> float | None:
    if not labels or not preds:
        return None
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_pred))


def run_training(args: argparse.Namespace) -> Path:
    train_data_path = os.environ.get("TRAIN_DATA_PATH")
    ckpt_dir = Path(os.environ.get("TRAIN_CKPT_PATH", "./ckpt"))
    log_dir = Path(os.environ.get("TRAIN_LOG_PATH", "./logs"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    files = find_parquet_files(train_data_path)
    if not files:
        raise FileNotFoundError(f"No parquet files found under TRAIN_DATA_PATH={train_data_path}")

    columns = read_parquet_columns(files[0])
    feature_config = build_feature_config(columns)
    if not feature_config.label_column:
        raise ValueError("No label column found. Expected one of label_type/label/target/is_conversion.")

    device = torch.device(args.device)
    model = TAAC2026Scorer(
        num_cat_fields=len(feature_config.cat_columns),
        num_dense_fields=len(feature_config.dense_columns),
        num_token_fields=len(feature_config.token_fields),
        hash_size=args.hash_size,
        dense_hash_size=args.dense_hash_size,
        hidden_dim=args.hidden_units,
        num_layers=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout_rate,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    log_path = log_dir / "taac2026_train.log"
    best_auc = -1.0
    best_path = ckpt_dir / "taac2026_pcvr.pt"
    global_step = 0
    start_time = time.time()

    with log_path.open("w", encoding="utf-8") as log_f:
        for epoch in range(1, args.num_epochs + 1):
            batcher = TAAC2026Batcher(
                train_data_path,
                feature_config,
                hash_size=args.hash_size,
                batch_size=args.batch_size,
                include_label=True,
            )
            model.train()
            train_loss = 0.0
            train_batches = 0
            val_labels: list[np.ndarray] = []
            val_preds: list[np.ndarray] = []

            for batch_idx, batch in enumerate(batcher):
                batch = move_batch_to_device(batch, device)
                labels = batch["labels"].float()

                if batch_idx < args.val_batches:
                    model.eval()
                    with torch.no_grad():
                        logits = model(
                            batch["cat_tokens"],
                            batch["dense_values"],
                            batch["dense_mask"],
                            batch["time_features"],
                        )
                    val_labels.append(labels.detach().cpu().numpy())
                    val_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
                    model.train()
                    continue

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(
                        batch["cat_tokens"],
                        batch["dense_values"],
                        batch["dense_mask"],
                        batch["time_features"],
                    )
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += float(loss.detach().cpu())
                train_batches += 1
                global_step += 1

                if global_step % 100 == 0:
                    msg = {
                        "epoch": epoch,
                        "step": global_step,
                        "loss": train_loss / max(train_batches, 1),
                        "elapsed": round(time.time() - start_time, 2),
                    }
                    log_f.write(json.dumps(msg) + "\n")
                    log_f.flush()

                if args.max_train_batches > 0 and train_batches >= args.max_train_batches:
                    break

            val_auc = compute_auc(val_labels, val_preds)
            mean_loss = train_loss / max(train_batches, 1)
            epoch_msg = {
                "epoch": epoch,
                "train_batches": train_batches,
                "loss": mean_loss,
                "val_auc": val_auc,
                "elapsed": round(time.time() - start_time, 2),
            }
            print(json.dumps(epoch_msg, ensure_ascii=False))
            log_f.write(json.dumps(epoch_msg) + "\n")
            log_f.flush()

            score_for_save = val_auc if val_auc is not None else -mean_loss
            if score_for_save > best_auc:
                best_auc = score_for_save
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "feature_config": feature_config.model_config,
                        "args": vars(args),
                        "val_auc": val_auc,
                    },
                    best_path,
                )

    return best_path


def main(argv: list[str] | None = None) -> None:
    args = get_args(argv)
    set_seed(args.seed)
    path = run_training(args)
    print(f"Saved TAAC2026 checkpoint: {path}")


if __name__ == "__main__":
    main()
