from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ID_COLUMNS = ["user_id", "item_id"]
TIME_COLUMNS = ["label_time", "timestamp"]
LABEL_COLUMNS = ["label_type", "label", "target", "is_conversion"]

USER_INT_SCALAR_IDS = [
    1,
    3,
    4,
    *range(48, 60),
    82,
    86,
    *range(92, 110),
]
USER_INT_LIST_IDS = [15, 60, 62, 63, 64, 65, 66, 80, 89, 90, 91]
USER_DENSE_IDS = [61, 62, 63, 64, 65, 66, 87, 89, 90, 91]

ITEM_INT_SCALAR_IDS = [5, 6, 7, 8, 9, 10, 12, 13, 16, 81, 83, 84, 85]
ITEM_INT_LIST_IDS = [11]

DOMAIN_SEQUENCE_IDS = [
    *range(38, 47),
    *range(67, 80),
    88,
    *range(27, 38),
    47,
    *range(17, 27),
]


def user_int_col(feat_id: int) -> str:
    return f"user_int_feats_{feat_id}"


def user_dense_col(feat_id: int) -> str:
    return f"user_dense_feats_{feat_id}"


def item_int_col(feat_id: int) -> str:
    return f"item_int_feats_{feat_id}"


def domain_seq_col(feat_id: int) -> str:
    if 38 <= feat_id <= 46:
        return f"domain_a_seq_{feat_id}"
    if 67 <= feat_id <= 79 or feat_id == 88:
        return f"domain_b_seq_{feat_id}"
    if 27 <= feat_id <= 37 or feat_id == 47:
        return f"domain_c_seq_{feat_id}"
    if 17 <= feat_id <= 26:
        return f"domain_d_seq_{feat_id}"
    raise ValueError(f"Unknown domain sequence feature id: {feat_id}")


EXPECTED_CAT_SCALAR_COLUMNS = [
    *ID_COLUMNS,
    *[user_int_col(i) for i in USER_INT_SCALAR_IDS],
    *[item_int_col(i) for i in ITEM_INT_SCALAR_IDS],
]
EXPECTED_LIST_COLUMNS = [
    *[user_int_col(i) for i in USER_INT_LIST_IDS],
    *[item_int_col(i) for i in ITEM_INT_LIST_IDS],
    *[domain_seq_col(i) for i in DOMAIN_SEQUENCE_IDS],
]
EXPECTED_DENSE_COLUMNS = [user_dense_col(i) for i in USER_DENSE_IDS]


@dataclass
class TAACFeatureConfig:
    scalar_columns: list[str]
    list_columns: list[str]
    dense_columns: list[str]
    label_column: str | None = None

    @property
    def cat_columns(self) -> list[str]:
        return self.scalar_columns + self.list_columns

    @property
    def token_fields(self) -> list[str]:
        return self.scalar_columns + self.list_columns + self.dense_columns + ["__time__"]

    @property
    def model_config(self) -> dict:
        return {
            "scalar_columns": self.scalar_columns,
            "list_columns": self.list_columns,
            "dense_columns": self.dense_columns,
            "label_column": self.label_column,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TAACFeatureConfig":
        return cls(
            scalar_columns=list(data.get("scalar_columns", [])),
            list_columns=list(data.get("list_columns", [])),
            dense_columns=list(data.get("dense_columns", [])),
            label_column=data.get("label_column"),
        )


def find_parquet_files(path: str | Path | None) -> list[Path]:
    if path is None:
        return []
    root = Path(path)
    if root.is_file() and root.suffix.lower() == ".parquet":
        return [root]
    if not root.exists() or not root.is_dir():
        return []
    direct = sorted(p for p in root.glob("*.parquet") if p.is_file())
    if direct:
        return direct
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def is_taac2026_data_dir(path: str | Path | None) -> bool:
    files = find_parquet_files(path)
    if not files:
        return False
    try:
        names = set(read_parquet_columns(files[0]))
    except Exception:
        return False
    required = {"user_id", "item_id"}
    feature_hint = any(name.startswith("user_int_feats_") for name in names)
    seq_hint = any(name.startswith("domain_") and "_seq_" in name for name in names)
    return required.issubset(names) and feature_hint and seq_hint


def read_parquet_columns(path: str | Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).schema_arrow.names


def build_feature_config(available_columns: Iterable[str]) -> TAACFeatureConfig:
    available = set(available_columns)
    label_column = next((col for col in LABEL_COLUMNS if col in available), None)
    return TAACFeatureConfig(
        scalar_columns=[col for col in EXPECTED_CAT_SCALAR_COLUMNS if col in available],
        list_columns=[col for col in EXPECTED_LIST_COLUMNS if col in available],
        dense_columns=[col for col in EXPECTED_DENSE_COLUMNS if col in available],
        label_column=label_column,
    )


def required_columns(config: TAACFeatureConfig, include_label: bool) -> list[str]:
    cols = [*config.scalar_columns, *config.list_columns, *config.dense_columns]
    for col in TIME_COLUMNS:
        if col not in cols:
            cols.append(col)
    if include_label and config.label_column and config.label_column not in cols:
        cols.append(config.label_column)
    return cols
