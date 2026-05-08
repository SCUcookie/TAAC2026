import json
import os
import traceback
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


MAX_SAMPLE_ROWS = 5
MAX_SAMPLE_VALUES = 10
MAX_STATS_ROWS = 5000
MAX_PRINT_FILES = 30
MAX_PRINT_COLUMNS = 200


def emit(line: str) -> None:
    print(line, flush=True)


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


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _list_relative_files(root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not root or not os.path.exists(root):
        return rows
    for current_root, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(current_root, name)
            rel = os.path.relpath(path, root)
            item: Dict[str, Any] = {"path": rel.replace("\\", "/")}
            try:
                item["size_bytes"] = os.path.getsize(path)
            except OSError as exc:
                item["size_bytes"] = None
                item["size_error"] = str(exc)
            rows.append(item)
    rows.sort(key=lambda x: x["path"])
    return rows


def _read_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_schema_preview(data_dir: str) -> Optional[Dict[str, Any]]:
    schema_path = os.path.join(data_dir, "schema.json")
    if not os.path.exists(schema_path):
        return None
    try:
        return _safe_jsonable(_read_json_file(schema_path))
    except Exception as exc:
        return {"error": str(exc)}


def _load_small_json_previews(data_dir: str) -> Dict[str, Any]:
    previews: Dict[str, Any] = {}
    for item in _list_relative_files(data_dir):
        rel = item["path"]
        if not rel.lower().endswith(".json"):
            continue
        full = os.path.join(data_dir, rel.replace("/", os.sep))
        try:
            previews[rel] = _safe_jsonable(_read_json_file(full))
        except Exception as exc:
            previews[rel] = {"error": str(exc)}
        if len(previews) >= 10:
            break
    return previews


def _sample_text_lines(path: str, max_lines: int = 3) -> List[str]:
    lines: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n")[:500])
    except Exception as exc:
        lines.append(f"<read_error: {exc}>")
    return lines


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
        result["flat_unique_sample"] = list(dict.fromkeys(str(x) for x in flat[:MAX_SAMPLE_VALUES]))
    return result


def _summarize_scalar(values: List[Any]) -> Dict[str, Any]:
    non_null = [v for v in values if v is not None]
    result: Dict[str, Any] = {
        "sample_values": _safe_jsonable(values[:MAX_SAMPLE_ROWS]),
        "non_null_count": len(non_null),
        "null_count": len(values) - len(non_null),
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
            result["approx_unique_count"] = len(counter)
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
    return {"sampled_rows": rows_seen, "column_stats": stats}


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

        if len(sampled_batches) < 5:
            for batch in pf.iter_batches(batch_size=1024):
                sampled_batches.append(batch)
                if len(sampled_batches) >= 5:
                    break

        if "user_id" in names:
            for batch in pf.iter_batches(columns=["user_id"], batch_size=8192):
                vals = batch.column(0).to_pylist()
                all_user_ids.extend(str(v) for v in vals if v is not None)

    sample_analysis = _collect_column_stats_from_batches(sampled_batches)
    summary = {
        "format": "parquet",
        "total_files": len(parquet_files),
        "total_rows": total_rows,
        "total_row_groups": total_row_groups,
        "columns": dataset_columns,
        "files": files_summary,
        "sample_analysis": sample_analysis,
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
            "sample_lines": _sample_text_lines(path),
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
    other_files: List[str] = []

    for current_root, _, files in os.walk(data_dir):
        for name in files:
            path = os.path.join(current_root, name)
            lower = name.lower()
            if lower.endswith(".parquet"):
                parquet_files.append(path)
            elif lower.endswith(".jsonl"):
                jsonl_files.append(path)
            else:
                other_files.append(path)

    parquet_files.sort()
    jsonl_files.sort()
    other_files.sort()

    if parquet_files:
        summary, user_ids = _summarize_parquet_dataset(data_dir, parquet_files)
    elif jsonl_files:
        summary, user_ids = _summarize_jsonl_dataset(data_dir, jsonl_files)
    else:
        summary, user_ids = ({
            "format": "unknown",
            "message": "No parquet or jsonl files found under EVAL_DATA_PATH.",
        }, [])

    summary["detected_file_types"] = {
        "parquet_count": len(parquet_files),
        "jsonl_count": len(jsonl_files),
        "other_count": len(other_files),
        "other_sample": [
            os.path.relpath(p, data_dir).replace("\\", "/")
            for p in other_files[:MAX_SAMPLE_VALUES]
        ],
    }
    return summary, user_ids


def _build_default_predictions(user_ids: List[str]) -> Dict[str, float]:
    return {uid: 0.5 for uid in user_ids}


def _print_dict_block(prefix: str, payload: Dict[str, Any]) -> None:
    emit(f"{prefix} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}")


def _print_dataset_summary(env_summary: Dict[str, Any], dataset_summary: Dict[str, Any]) -> None:
    emit("===== DATASET PROBE SUMMARY BEGIN =====")
    _print_dict_block("ENV_SUMMARY", env_summary)

    compact_summary = {
        "format": dataset_summary.get("format"),
        "message": dataset_summary.get("message"),
        "total_files": dataset_summary.get("total_files"),
        "total_rows": dataset_summary.get("total_rows"),
        "total_row_groups": dataset_summary.get("total_row_groups"),
        "user_id_count": dataset_summary.get("user_id_count"),
        "user_id_sample": dataset_summary.get("user_id_sample"),
        "detected_file_types": dataset_summary.get("detected_file_types"),
    }
    _print_dict_block("DATASET_SUMMARY", compact_summary)

    columns = dataset_summary.get("columns", {})
    emit(f"COLUMN_COUNT {len(columns)}")
    for idx, name in enumerate(sorted(columns.keys())):
        if idx >= MAX_PRINT_COLUMNS:
            emit(f"COLUMN_TRUNCATED remaining={len(columns) - MAX_PRINT_COLUMNS}")
            break
        emit(f"COLUMN {name} :: {columns[name]}")

    for item in dataset_summary.get("files", [])[:MAX_PRINT_FILES]:
        _print_dict_block("FILE", item)

    inventory = dataset_summary.get("file_inventory", [])
    emit(f"INVENTORY_COUNT {len(inventory)}")
    for item in inventory[:MAX_PRINT_FILES]:
        _print_dict_block("INVENTORY", item)

    schema_preview = dataset_summary.get("schema_json_preview")
    if schema_preview is not None:
        _print_dict_block("SCHEMA_PREVIEW", {"schema": schema_preview})

    json_previews = dataset_summary.get("json_file_previews")
    if json_previews:
        _print_dict_block("JSON_FILE_PREVIEWS", json_previews)

    sample_analysis = dataset_summary.get("sample_analysis", {})
    emit(f"SAMPLED_ROWS {sample_analysis.get('sampled_rows')}")
    column_stats = sample_analysis.get("column_stats", {})
    for idx, name in enumerate(sorted(column_stats.keys())):
        if idx >= MAX_PRINT_COLUMNS:
            emit(f"STATS_TRUNCATED remaining={len(column_stats) - MAX_PRINT_COLUMNS}")
            break
        _print_dict_block("STATS", {"column": name, "stats": column_stats[name]})

    emit("===== DATASET PROBE SUMMARY END =====")


def _fallback_summary(data_dir: str, env_summary: Dict[str, Any], exc: Exception) -> Tuple[Dict[str, Any], List[str]]:
    return {
        "format": "probe_error",
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "file_inventory": _list_relative_files(data_dir),
        "env_summary": env_summary,
    }, []


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
        "cwd": os.getcwd(),
        "EVAL_DATA_PATH": data_dir,
        "EVAL_RESULT_PATH": result_dir,
        "EVAL_INFER_PATH": infer_dir,
        "MODEL_OUTPUT_PATH": model_dir,
        "USER_CACHE_PATH": cache_dir,
    }

    emit("Starting dataset probe")
    emit(f"EVAL_DATA_PATH={data_dir}")

    try:
        dataset_summary, user_ids = _discover_dataset(data_dir)
        dataset_summary["schema_json_preview"] = _load_schema_preview(data_dir)
        dataset_summary["file_inventory"] = _list_relative_files(data_dir)
        dataset_summary["json_file_previews"] = _load_small_json_previews(data_dir)
    except Exception as exc:
        emit(f"PROBE_EXCEPTION {exc}")
        emit(traceback.format_exc())
        dataset_summary, user_ids = _fallback_summary(data_dir, env_summary, exc)

    _print_dataset_summary(env_summary, dataset_summary)

    if not user_ids:
        emit("WARNING no user_id values discovered; predictions.json will be empty.")
    predictions = {"predictions": _build_default_predictions(user_ids)}

    _write_json(os.path.join(result_dir, "platform_env.json"), env_summary)
    _write_json(os.path.join(result_dir, "dataset_summary.json"), dataset_summary)
    _write_json(os.path.join(result_dir, "predictions.json"), predictions)

    emit("WROTE platform_env.json")
    emit("WROTE dataset_summary.json")
    emit(f"WROTE predictions.json user_id_count={len(user_ids)}")


if __name__ == "__main__":
    main()
