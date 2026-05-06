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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "as_py"):
        value = value.as_py()
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
    if math.isfinite(out):
        return out
    return 0.0


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
    def __init__(
        self,
        data_path: str | Path,
        config: TAACFeatureConfig,
        hash_size: int,
        batch_size: int,
        include_label: bool,
    ) -> None:
        self.files = find_parquet_files(data_path)
        if not self.files:
            raise FileNotFoundError(f"No parquet files found under {data_path}")
        self.config = config
        self.hash_size = hash_size
        self.batch_size = batch_size
        self.include_label = include_label
        self.columns = required_columns(config, include_label)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | list]]:
        for file_path in self.files:
            parquet_file = pq.ParquetFile(file_path)
            columns = [c for c in self.columns if c in parquet_file.schema_arrow.names]
            for record_batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=columns):
                table = record_batch.to_pydict()
                size = record_batch.num_rows
                yield self._convert_batch(table, size)

    def _convert_batch(self, rows: dict[str, list], size: int) -> dict[str, torch.Tensor | list]:
        cat = np.zeros((size, len(self.config.cat_columns)), dtype=np.int64)
        for col_idx, col in enumerate(self.config.scalar_columns):
            values = rows.get(col)
            if values is None:
                continue
            for row_idx, value in enumerate(values):
                cat[row_idx, col_idx] = stable_hash(f"{col}={value}", self.hash_size)

        offset = len(self.config.scalar_columns)
        for list_idx, col in enumerate(self.config.list_columns):
            values = rows.get(col)
            if values is None:
                continue
            for row_idx, value in enumerate(values):
                hashed = [stable_hash(f"{col}={v}", self.hash_size) for v in _as_list(value)]
                cat[row_idx, offset + list_idx] = max(hashed) if hashed else 0

        dense = np.zeros((size, len(self.config.dense_columns)), dtype=np.float32)
        dense_mask = np.zeros_like(dense, dtype=np.bool_)
        for col_idx, col in enumerate(self.config.dense_columns):
            values = rows.get(col)
            if values is None:
                continue
            for row_idx, value in enumerate(values):
                if _is_missing(value):
                    continue
                dense[row_idx, col_idx] = _numeric_value(value)
                dense_mask[row_idx, col_idx] = True

        batch: dict[str, torch.Tensor | list] = {
            "cat_tokens": torch.from_numpy(cat),
            "dense_values": torch.from_numpy(dense),
            "dense_mask": torch.from_numpy(dense_mask),
            "time_features": torch.from_numpy(build_time_features(rows, size)),
        }

        if self.include_label and self.config.label_column:
            labels = np.asarray(rows.get(self.config.label_column, np.zeros(size)), dtype=np.float32)
            if self.config.label_column == "label_type":
                labels = (labels > 0).astype(np.float32)
            batch["labels"] = torch.from_numpy(labels)

        user_ids = rows.get("user_id")
        item_ids = rows.get("item_id")
        if user_ids is not None:
            batch["user_id"] = [str(x) for x in user_ids]
        if item_ids is not None:
            batch["item_id"] = [str(x) for x in item_ids]
        return batch


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
