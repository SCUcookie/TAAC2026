from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch

from taac2026_schema import TAACFeatureConfig, find_parquet_files, required_columns


def stable_hash(value: object, bucket_size: int) -> int:
    if value is None:
        return 0
    text = str(value)
    if not text or text.lower() in {"nan", "none", "null"}:
        return 0
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % bucket_size + 1


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, np.ndarray)):
        return False
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _as_list(value: object) -> list:
    if _is_missing(value):
        return []
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _numeric_value(value: object) -> float:
    if _is_missing(value):
        return 0.0
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = [_numeric_value(v) for v in list(value)]
        return float(np.mean(vals)) if vals else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = 0.0
    return out if math.isfinite(out) else 0.0


def build_time_features(rows: dict[str, object], size: int) -> np.ndarray:
    label_time = rows.get("label_time")
    timestamp = rows.get("timestamp")
    out = np.zeros((size, 6), dtype=np.float32)

    label_arr = np.asarray(label_time if label_time is not None else np.zeros(size), dtype=np.float64)
    ts_arr = np.asarray(timestamp if timestamp is not None else np.zeros(size), dtype=np.float64)
    if label_arr.shape[0] != size:
        label_arr = np.zeros(size, dtype=np.float64)
    if ts_arr.shape[0] != size:
        ts_arr = np.zeros(size, dtype=np.float64)

    base = np.where(label_arr > 0, label_arr, ts_arr)
    safe_base = np.where(base > 0, base, 0.0)
    hours = np.mod(np.floor_divide(safe_base.astype(np.int64), 3600), 24)
    weekdays = np.mod(np.floor_divide(safe_base.astype(np.int64), 86400), 7)
    out[:, 0] = np.sin(2 * np.pi * hours / 24.0)
    out[:, 1] = np.cos(2 * np.pi * hours / 24.0)
    out[:, 2] = np.sin(2 * np.pi * weekdays / 7.0)
    out[:, 3] = np.cos(2 * np.pi * weekdays / 7.0)

    delta = np.maximum(label_arr - ts_arr, 0.0)
    out[:, 4] = np.log1p(delta) / 20.0
    out[:, 5] = (safe_base > 0).astype(np.float32)
    return out


class TAAC2026Batcher:
    """Streaming Parquet loader shaped like the official PCVRHyFormer baseline."""

    def __init__(
        self,
        data_path: str | Path,
        config: TAACFeatureConfig,
        hash_size: int,
        batch_size: int,
        include_label: bool,
        seq_len: int = 32,
    ) -> None:
        self.files = find_parquet_files(data_path)
        if not self.files:
            raise FileNotFoundError(f"No parquet files found under {data_path}")
        self.config = config
        self.hash_size = hash_size
        self.batch_size = batch_size
        self.include_label = include_label
        self.seq_len = seq_len
        self.columns = required_columns(config, include_label)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | list]]:
        for file_path in self.files:
            parquet_file = pq.ParquetFile(file_path)
            columns = [c for c in self.columns if c in parquet_file.schema_arrow.names]
            for record_batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=columns):
                rows = record_batch.to_pydict()
                yield self._convert_batch(rows, record_batch.num_rows)

    def _column_token(self, col: str, value: object) -> int:
        values = _as_list(value)
        if not values:
            return 0
        hashes = [stable_hash(f"{col}={v}", self.hash_size) for v in values]
        return max(hashes) if hashes else 0

    def _convert_batch(self, rows: dict[str, list], size: int) -> dict[str, torch.Tensor | list]:
        ns_tokens = np.zeros((size, self.config.ns_token_count), dtype=np.int64)
        ns_dense = np.zeros((size, self.config.ns_token_count), dtype=np.float32)
        time_features = build_time_features(rows, size)

        for group_idx, (group_name, columns) in enumerate(self.config.ns_groups.items()):
            for row_idx in range(size):
                cat_hashes = []
                dense_values = []
                for col in columns:
                    if col == "__time__":
                        dense_values.extend(time_features[row_idx].tolist())
                        continue
                    values = rows.get(col)
                    if values is None:
                        continue
                    value = values[row_idx]
                    if col in self.config.dense_columns:
                        dense_values.append(_numeric_value(value))
                    else:
                        token = self._column_token(col, value)
                        if token:
                            cat_hashes.append(token)
                if cat_hashes:
                    ns_tokens[row_idx, group_idx] = max(cat_hashes)
                if dense_values:
                    ns_dense[row_idx, group_idx] = float(np.mean(dense_values))

        seq_tokens = np.zeros((size, self.config.seq_token_count, self.seq_len), dtype=np.int64)
        seq_mask = np.zeros((size, self.config.seq_token_count, self.seq_len), dtype=np.bool_)
        for domain_idx, (_domain_name, columns) in enumerate(self.config.seq_groups.items()):
            for row_idx in range(size):
                domain_values = []
                for col in columns:
                    values = rows.get(col)
                    if values is None:
                        continue
                    for value in _as_list(values[row_idx]):
                        token = stable_hash(f"{col}={value}", self.hash_size)
                        if token:
                            domain_values.append(token)
                if not domain_values:
                    continue
                clipped = domain_values[-self.seq_len :]
                start = self.seq_len - len(clipped)
                seq_tokens[row_idx, domain_idx, start:] = np.asarray(clipped, dtype=np.int64)
                seq_mask[row_idx, domain_idx, start:] = True

        batch: dict[str, torch.Tensor | list] = {
            "ns_tokens": torch.from_numpy(ns_tokens),
            "ns_dense": torch.from_numpy(ns_dense),
            "seq_tokens": torch.from_numpy(seq_tokens),
            "seq_mask": torch.from_numpy(seq_mask),
            "time_features": torch.from_numpy(time_features),
        }

        if self.include_label and self.config.label_column:
            labels = np.asarray(rows.get(self.config.label_column, np.zeros(size)), dtype=np.float32)
            if self.config.label_column == "label_type":
                labels = (labels > 0).astype(np.float32)
            batch["labels"] = torch.from_numpy(labels)

        if "user_id" in rows:
            batch["user_id"] = [str(x) for x in rows["user_id"]]
        if "item_id" in rows:
            batch["item_id"] = [str(x) for x in rows["item_id"]]
        return batch


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
