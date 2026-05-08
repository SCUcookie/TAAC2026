from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from taac2026_data import TAAC2026Batcher, move_batch_to_device
from taac2026_model import PCVRHyFormer
from taac2026_schema import build_feature_config, dump_ns_groups, find_parquet_files, read_parquet_columns


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
    parser.add_argument("--val_strategy", choices=["prefix", "tail"], default="tail")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--embedding_lr", type=float, default=1e-3)
    parser.add_argument("--pairwise_auc_weight", type=float, default=0.05)
    parser.add_argument("--pairwise_max_pairs", type=int, default=8192)
    parser.add_argument("--focal_gamma", type=float, default=0.0)
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


def compute_logloss(labels: list[np.ndarray], preds: list[np.ndarray]) -> float | None:
    if not labels or not preds:
        return None
    y_true = np.concatenate(labels).astype(np.float64)
    y_pred = np.clip(np.concatenate(preds).astype(np.float64), 1e-7, 1.0 - 1e-7)
    if y_true.size == 0:
        return None
    loss = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return float(np.mean(loss))


def bce_with_optional_focal(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if gamma <= 0:
        return bce.mean()
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        pt = torch.where(labels > 0.5, probs, 1.0 - probs).clamp(1e-6, 1.0)
        focal_weight = torch.pow(1.0 - pt, gamma)
    return (focal_weight * bce).mean()


def pairwise_auc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_pairs: int,
) -> torch.Tensor:
    positives = logits[labels > 0.5]
    negatives = logits[labels <= 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return logits.new_zeros(())

    num_pairs = positives.numel() * negatives.numel()
    if max_pairs > 0 and num_pairs > max_pairs:
        pos_idx = torch.randint(positives.numel(), (max_pairs,), device=logits.device)
        neg_idx = torch.randint(negatives.numel(), (max_pairs,), device=logits.device)
        diffs = positives[pos_idx] - negatives[neg_idx]
    else:
        diffs = positives[:, None] - negatives[None, :]
    return torch.nn.functional.softplus(-diffs).mean()


def batch_forward(model: PCVRHyFormer, batch: dict) -> torch.Tensor:
    return model(
        batch["ns_tokens"],
        batch["ns_dense"],
        batch["seq_tokens"],
        batch["seq_mask"],
        batch["time_features"],
    )


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
    dump_ns_groups(feature_config, ckpt_dir / "ns_groups.json")

    model = PCVRHyFormer(
        num_ns_groups=feature_config.ns_token_count,
        num_seq_domains=feature_config.seq_token_count,
        hash_size=args.hash_size,
        hidden_dim=args.hidden_units,
        num_layers=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout_rate,
    ).to(device)

    emb_params = []
    main_params = []
    for name, param in model.named_parameters():
        if "embedding" in name:
            emb_params.append(param)
        else:
            main_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": main_params, "lr": args.lr},
            {"params": emb_params, "lr": args.embedding_lr},
        ],
        weight_decay=args.weight_decay,
    )
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
                seq_len=args.seq_len,
            )
            model.train()
            train_loss = 0.0
            train_batches = 0
            val_batches: deque[dict] | None = deque(maxlen=args.val_batches) if args.val_strategy == "tail" else None
            val_labels: list[np.ndarray] = []
            val_preds: list[np.ndarray] = []

            for batch_idx, batch in enumerate(batcher):
                batch = move_batch_to_device(batch, device)
                labels = batch["labels"].float()

                if args.val_strategy == "prefix" and batch_idx < args.val_batches:
                    model.eval()
                    with torch.no_grad():
                        logits = batch_forward(model, batch)
                    val_labels.append(labels.detach().cpu().numpy())
                    val_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
                    model.train()
                    continue

                if args.val_strategy == "tail" and args.val_batches > 0:
                    if val_batches is not None and len(val_batches) == args.val_batches:
                        train_batch = val_batches.popleft()
                    else:
                        train_batch = None
                    val_batches.append(batch)
                    if train_batch is None:
                        continue
                    batch = train_batch
                    labels = batch["labels"].float()

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = batch_forward(model, batch)
                    loss = bce_with_optional_focal(logits, labels, args.focal_gamma)
                    if args.pairwise_auc_weight > 0:
                        pair_loss = pairwise_auc_loss(logits, labels, args.pairwise_max_pairs)
                        loss = loss + args.pairwise_auc_weight * pair_loss
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

            if args.val_strategy == "tail" and val_batches:
                model.eval()
                with torch.no_grad():
                    for val_batch in val_batches:
                        val_batch = move_batch_to_device(val_batch, device)
                        labels = val_batch["labels"].float()
                        logits = batch_forward(model, val_batch)
                        val_labels.append(labels.detach().cpu().numpy())
                        val_preds.append(torch.sigmoid(logits).detach().cpu().numpy())

            val_auc = compute_auc(val_labels, val_preds)
            val_logloss = compute_logloss(val_labels, val_preds)
            mean_loss = train_loss / max(train_batches, 1)
            epoch_msg = {
                "epoch": epoch,
                "train_batches": train_batches,
                "loss": mean_loss,
                "val_auc": val_auc,
                "val_logloss": val_logloss,
                "val_strategy": args.val_strategy,
                "pairwise_auc_weight": args.pairwise_auc_weight,
                "focal_gamma": args.focal_gamma,
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
