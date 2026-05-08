from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate platform inference outputs before spending an evaluation slot."
    )
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default="predictions.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_path = result_dir / args.output_name
    if not output_path.exists():
        raise FileNotFoundError(f"Missing output file: {output_path}")

    with output_path.open("r", encoding="utf-8") as f:
        payload: Dict[str, Any] = json.load(f)

    if "predictions" not in payload or not isinstance(payload["predictions"], dict):
        raise ValueError("predictions.json must contain a top-level 'predictions' object")

    predictions = payload["predictions"]
    if not predictions:
        raise ValueError("predictions.json is empty")

    bad_key_types = [k for k in predictions.keys() if not isinstance(k, str)]
    if bad_key_types:
        raise ValueError(f"Found non-string prediction keys, sample={bad_key_types[:5]}")

    bad_values = []
    for user_id, score in predictions.items():
        if not isinstance(score, (int, float)):
            bad_values.append((user_id, type(score).__name__))
            continue
        if score < 0:
            bad_values.append((user_id, score))
    if bad_values:
        raise ValueError(f"Found invalid scores, sample={bad_values[:5]}")

    sample_items = list(predictions.items())[:5]
    summary = {
        "output_path": str(output_path),
        "num_predictions": len(predictions),
        "sample_predictions": sample_items,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
