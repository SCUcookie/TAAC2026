import json
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


MAX_SAMPLE_ROWS = 5
MAX_SAMPLE_VALUES = 10
MAX_STATS_ROWS = 5000


def _safe_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_jsonable(v) for v in value[:MAX_SAMPLE_VALUES]]
    if isinstance(value, tuple):
        return [_safe_jsonable(v) for v in list(value)[:MAX_SAMPLE_VALUES]]
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    return str(value)


def _list_relative_files(root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for current_root, _, files in os.walk(root):
        for name in sorted(files):
            path = os.path.join(current_root, name)
            rel = os.path.relpath(path, root)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = None
            rows.append({"path": rel.replace("\\", "/"), "size_bytes": size})
    rows.sort(key=lambda x: x["path"])
    return rows


def _load_schema_preview(data_dir: str) -> Optional[Dict[str, Any]]:
    schema_path = os.path.join(data_dir, "schema.json")
    if not os.path.exists(schema_path):
        return None
    with open(schema_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return _safe_jsonable(raw)


def _summarize_sequence(values: List[Any]) -> Dict[str, Any]:
    lengths = [len(v) for v in values if isinstance(v, list)]
    flat = []
    for v in values:
        if isinstance(v, list):
            flat.extend(v[:MAX_SAMPLE_VALUES])
    result: Dict[str, Any] = {
        "sample_values": _safe_jsonable(values[:MAX_SAMPLE_ROWS]),
        "non_null_count": sum(v is not None for v in values),
        "list_count": len(lengths),
    }
    if lengths:
        result["length_stats"] = {
            "min": min(lengths),
            "max": max(lengths),
            "avg": round(sum(lengths) / len(lengths), 4),
        }
    if flat:
        numeric_flat = [x for x in flat if isinstance(x, (int, float))]
        if numeric_flat:
            result["flat_value_stats"] = {
                "min": min(numeric_flat),
                "max": max(numeric_flat),
            }
    return result


def _summarize_scalar(values: List[Any]) -> Dict[str, Any]:
    non_null = [v for v in values if v is not None]
    result: Dict[str, Any] = {
        "sample_values": _safe_jsonable(values[:MAX_SAMPLE_ROWS]),
        "non_null_count": len(non_null),
    }
    if non_null and all(isinstance(v, (int, float)) for v in non_null):
        result["value_stats"] = {
            "min": min(non_null),
            "max": max(non_null),
            "avg": round(sum(non_null) / len(non_null), 6),
        }
    else:
        counter = Counter(str(v) for v in non_null[:MAX_STATS_ROWS])
        if counter:
            result["top_values"] = counter.most_common(MAX_SAMPLE_VALUES)
    return result


def _extract_pylist(column: Any) -> List[Any]:
    try:
        return column.to_pylist()
    except Exception:
        return [str(x) for x in column]


def _collect_column_stats_from_batches(batches: Iterable[Any]) -> Dict[str, Any]:
    column_values: Dict[str, List[Any]] = {}
    rows_seen = 0

    for batch in batches:
        names = list(batch.schema.names)
        take_rows = min(batch.num_rows, max(0, MAX_STATS_ROWS - rows_seen))
        if take_rows <= 0:
            break
        for idx, name in enumerate(names):
            values = _extract_pylist(batch.column(idx))[:take_rows]
            column_values.setdefault(name, []).extend(values)
        rows_seen += take_rows
        if rows_seen >= MAX_STATS_ROWS:
            break

    stats: Dict[str, Any] = {}
    for name, values in column_values.items():
        if any(isinstance(v, list) for v in values if v is not None):
            stats[name] = _summarize_sequence(values)
        else:
            stats[name] = _summarize_scalar(values)
    return {
        "sampled_rows": rows_seen,
        "column_stats": stats,
    }


def _read_all_user_ids_from_parquet(parquet_files: List[str]) -> List[str]:
    import pyarrow.parquet as pq

    user_ids: List[str] = []
    for path in parquet_files:
        pf = pq.ParquetFile(path)
        if "user_id" not in pf.schema_arrow.names:
            continue
        for batch in pf.iter_batches(columns=["user_id"], batch_size=8192):
            values = batch.column(0).to_pylist()
            user_ids.extend(str(v) for v in values if v is not None)
    return user_ids


def _summarize_parquet_dataset(data_dir: str, parquet_files: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    import pyarrow.parquet as pq

    files_summary: List[Dict[str, Any]] = []
    all_user_ids: List[str] = []
    dataset_columns: Dict[str, str] = {}
    sampled_batches = []
    total_rows = 0
    total_row_groups = 0

    for path in parquet_files:
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        types = [str(field.type) for field in pf.schema_arrow]
        file_rows = pf.metadata.num_rows
        file_row_groups = pf.metadata.num_row_groups
        total_rows += file_rows
        total_row_groups += file_row_groups
        files_summary.append({
            "path": os.path.relpath(path, data_dir).replace("\\", "/"),
            "num_rows": file_rows,
            "num_row_groups": file_row_groups,
            "columns": names,
        })
        for name, typ in zip(names, types):
            dataset_columns.setdefault(name, typ)

        if len(sampled_batches) < 3:
            for batch in pf.iter_batches(batch_size=1024):
                sampled_batches.append(batch)
                if len(sampled_batches) >= 3:
                    break

        if "user_id" in names:
            for batch in pf.iter_batches(columns=["user_id"], batch_size=8192):
                vals = batch.column(0).to_pylist()
                all_user_ids.extend(str(v) for v in vals if v is not None)

    sampled = _collect_column_stats_from_batches(sampled_batches)
    summary = {
        "format": "parquet",
        "total_files": len(parquet_files),
        "total_rows": total_rows,
        "total_row_groups": total_row_groups,
        "columns": dataset_columns,
        "files": files_summary,
        "sample_analysis": sampled,
        "user_id_count": len(all_user_ids),
        "user_id_sample": all_user_ids[:MAX_SAMPLE_VALUES],
    }
    return summary, all_user_ids


def _iter_jsonl_rows(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _summarize_jsonl_dataset(data_dir: str, jsonl_files: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    file_summaries: List[Dict[str, Any]] = []
    field_samples: Dict[str, List[Any]] = {}
    user_ids: List[str] = []
    total_rows = 0

    for path in jsonl_files:
        rows_in_file = 0
        for row in _iter_jsonl_rows(path):
            rows_in_file += 1
            total_rows += 1
            uid = row.get("user_id")
            if uid is not None:
                user_ids.append(str(uid))
            if total_rows <= MAX_STATS_ROWS:
                for key, value in row.items():
                    field_samples.setdefault(key, []).append(value)
        file_summaries.append({
            "path": os.path.relpath(path, data_dir).replace("\\", "/"),
            "num_rows": rows_in_file,
        })

    columns = {k: type(v[0]).__name__ for k, v in field_samples.items() if v}
    stats: Dict[str, Any] = {}
    for key, values in field_samples.items():
        if any(isinstance(v, list) for v in values if v is not None):
            stats[key] = _summarize_sequence(values)
        else:
            stats[key] = _summarize_scalar(values)

    summary = {
        "format": "jsonl",
        "total_files": len(jsonl_files),
        "total_rows": total_rows,
        "columns": columns,
        "files": file_summaries,
        "sample_analysis": {
            "sampled_rows": min(total_rows, MAX_STATS_ROWS),
            "column_stats": stats,
        },
        "user_id_count": len(user_ids),
        "user_id_sample": user_ids[:MAX_SAMPLE_VALUES],
    }
    return summary, user_ids


def _discover_dataset(data_dir: str) -> Tuple[Dict[str, Any], List[str]]:
    parquet_files: List[str] = []
    jsonl_files: List[str] = []

    for current_root, _, files in os.walk(data_dir):
        for name in files:
            path = os.path.join(current_root, name)
            lower = name.lower()
            if lower.endswith(".parquet"):
                parquet_files.append(path)
            elif lower.endswith(".jsonl"):
                jsonl_files.append(path)

    parquet_files.sort()
    jsonl_files.sort()

    if parquet_files:
        return _summarize_parquet_dataset(data_dir, parquet_files)
    if jsonl_files:
        return _summarize_jsonl_dataset(data_dir, jsonl_files)
    return {
        "format": "unknown",
        "message": "No parquet or jsonl files found under EVAL_DATA_PATH.",
    }, []


def _build_default_predictions(user_ids: List[str]) -> Dict[str, float]:
    return {uid: 0.5 for uid in user_ids}


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_text_summary(path: str, env_summary: Dict[str, Any], dataset_summary: Dict[str, Any]) -> None:
    lines = [
        "Platform Dataset Probe Summary",
        "",
        "[Environment]",
    ]
    for key, value in env_summary.items():
        lines.append(f"{key}: {value}")

    lines.extend([
        "",
        "[Dataset]",
        f"format: {dataset_summary.get('format')}",
        f"message: {dataset_summary.get('message', '')}",
        f"total_files: {dataset_summary.get('total_files', '')}",
        f"total_rows: {dataset_summary.get('total_rows', '')}",
        f"user_id_count: {dataset_summary.get('user_id_count', '')}",
        "",
        "[Columns]",
    ])

    for name, typ in sorted(dataset_summary.get("columns", {}).items()):
        lines.append(f"{name}: {typ}")

    lines.extend([
        "",
        "[Files]",
    ])
    for item in dataset_summary.get("files", []):
        lines.append(json.dumps(item, ensure_ascii=False))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _print_dataset_summary(env_summary: Dict[str, Any], dataset_summary: Dict[str, Any]) -> None:
    print("===== DATASET PROBE SUMMARY BEGIN =====", flush=True)
    for key, value in env_summary.items():
        print(f"ENV {key}={value}", flush=True)

    print(f"DATASET format={dataset_summary.get('format')}", flush=True)
    if dataset_summary.get("message"):
        print(f"DATASET message={dataset_summary.get('message')}", flush=True)
    if dataset_summary.get("total_files") is not None:
        print(f"DATASET total_files={dataset_summary.get('total_files')}", flush=True)
    if dataset_summary.get("total_rows") is not None:
        print(f"DATASET total_rows={dataset_summary.get('total_rows')}", flush=True)
    if dataset_summary.get("total_row_groups") is not None:
        print(f"DATASET total_row_groups={dataset_summary.get('total_row_groups')}", flush=True)
    if dataset_summary.get("user_id_count") is not None:
        print(f"DATASET user_id_count={dataset_summary.get('user_id_count')}", flush=True)
    if dataset_summary.get("user_id_sample"):
        print(f"DATASET user_id_sample={dataset_summary.get('user_id_sample')}", flush=True)

    columns = dataset_summary.get("columns", {})
    print(f"DATASET column_count={len(columns)}", flush=True)
    for name, typ in sorted(columns.items()):
        print(f"COLUMN {name} :: {typ}", flush=True)

    for item in dataset_summary.get("files", [])[:10]:
        print(
            f"FILE path={item.get('path')} rows={item.get('num_rows')} "
            f"row_groups={item.get('num_row_groups')}",
            flush=True,
        )

    inventory = dataset_summary.get("file_inventory", [])
    print(f"DATASET inventory_count={len(inventory)}", flush=True)
    for item in inventory[:20]:
        print(
            f"INVENTORY path={item.get('path')} size_bytes={item.get('size_bytes')}",
            flush=True,
        )

    schema_preview = dataset_summary.get("schema_json_preview")
    if schema_preview is not None:
        print(f"SCHEMA preview={schema_preview}", flush=True)

    sample_analysis = dataset_summary.get("sample_analysis", {})
    print(f"SAMPLE sampled_rows={sample_analysis.get('sampled_rows')}", flush=True)
    column_stats = sample_analysis.get("column_stats", {})
    for name in sorted(column_stats.keys()):
        stats = column_stats[name]
        print(f"STATS {name} :: {stats}", flush=True)

    print("===== DATASET PROBE SUMMARY END =====", flush=True)


def main() -> None:
    data_dir = os.environ.get("EVAL_DATA_PATH", "")
    result_dir = os.environ.get("EVAL_RESULT_PATH", "")
    infer_dir = os.environ.get("EVAL_INFER_PATH", "")
    model_dir = os.environ.get("MODEL_OUTPUT_PATH", "")
    cache_dir = os.environ.get("USER_CACHE_PATH", "")

    if not result_dir:
        raise RuntimeError("EVAL_RESULT_PATH is not set.")
    os.makedirs(result_dir, exist_ok=True)

    env_summary = {
        "EVAL_DATA_PATH": data_dir,
        "EVAL_RESULT_PATH": result_dir,
        "EVAL_INFER_PATH": infer_dir,
        "MODEL_OUTPUT_PATH": model_dir,
        "USER_CACHE_PATH": cache_dir,
    }
    print("Starting dataset probe", flush=True)
    print(f"EVAL_DATA_PATH={data_dir}", flush=True)

    dataset_summary, user_ids = _discover_dataset(data_dir)
    dataset_summary["schema_json_preview"] = _load_schema_preview(data_dir)
    dataset_summary["file_inventory"] = _list_relative_files(data_dir)
    _print_dataset_summary(env_summary, dataset_summary)

    if not user_ids:
        print("WARNING no user_id values discovered; predictions.json will be empty.", flush=True)
    predictions = {"predictions": _build_default_predictions(user_ids)}

    _write_json(os.path.join(result_dir, "platform_env.json"), env_summary)
    _write_json(os.path.join(result_dir, "dataset_summary.json"), dataset_summary)
    _write_json(os.path.join(result_dir, "predictions.json"), predictions)
    _write_text_summary(
        os.path.join(result_dir, "dataset_summary.txt"),
        env_summary,
        dataset_summary,
    )

    print("WROTE platform_env.json", flush=True)
    print("WROTE dataset_summary.json", flush=True)
    print(f"WROTE predictions.json user_id_count={len(user_ids)}", flush=True)


if __name__ == "__main__":
    main()
