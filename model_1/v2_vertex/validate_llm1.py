import argparse
import json
import math
import os
from pathlib import Path

from Levenshtein import ratio as similarity


def find_ground_truth_root(start: Path) -> Path:
    """Locate data_pipeline/data-pipeline folder starting from current file."""
    for parent in [start] + list(start.parents):
        candidate = parent / "data_pipeline"
        if candidate.exists():
            return candidate / "data" / "unstructured" / "invoices"
        alt = parent / "data-pipeline"
        if alt.exists():
            return alt / "data" / "unstructured" / "invoices"
    return start


ROOT = Path(__file__).resolve().parent
OUTPUT_FOLDER = ROOT / "output"
GROUND_TRUTH_FOLDER = find_ground_truth_root(ROOT)
SAVE_DIR = ROOT / "eval_simple"
SAVE_DIR.mkdir(exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_filename(origin: str) -> str:
    base = os.path.basename(origin)
    return base.replace(".pdf", "")


def flatten_json(data, parent: str = "", sep: str = "."):
    """Flatten nested dict/list structures for easier comparison."""
    items = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent}{sep}{key}" if parent else key
            if isinstance(value, (dict, list)):
                items.update(flatten_json(value, new_key, sep))
            else:
                items[new_key] = value
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            new_key = f"{parent}{sep}{idx}" if parent else str(idx)
            if isinstance(value, (dict, list)):
                items.update(flatten_json(value, new_key, sep))
            else:
                items[new_key] = value
    else:
        items[parent or "value"] = data
    return items


def simple_match(a, b):
    if a == b:
        return True

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0.01)

    if isinstance(a, str) and isinstance(b, str):
        t, p = a.strip().lower(), b.strip().lower()
        if t == p:
            return True
        if similarity(t, p) > 0.80:
            return "partial"

    return False


def evaluate_prediction(out_file: Path, truth_dir: Path):
    pred_json = load_json(out_file)

    origin_file = pred_json.get("origin_file")
    if not origin_file:
        print(f"⚠ Skipping {out_file} (no origin_file)")
        return None

    base_name = extract_filename(origin_file)
    truth_path = truth_dir / f"{base_name}.json"

    if not truth_path.exists():
        print(f"⚠ No ground truth for {base_name}. Skipping.")
        return None

    truth_json = load_json(truth_path)["0_expected"]

    truth_flat = flatten_json(truth_json)
    pred_flat = flatten_json(pred_json)

    TP = FP = FN = 0
    wrong_fields = []
    partial_fields = []
    missing_fields = []
    extra_fields = []

    truth_keys = set(truth_flat.keys())
    pred_keys = set(pred_flat.keys())

    for key in truth_keys:
        if key in pred_keys:
            match = simple_match(truth_flat[key], pred_flat[key])
            if match is True:
                TP += 1
            elif match == "partial":
                FP += 1
                partial_fields.append(key)
            else:
                FP += 1
                wrong_fields.append(key)
        else:
            FN += 1
            missing_fields.append(key)

    for key in pred_keys - truth_keys:
        extra_fields.append(key)

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    report = {
        "invoice": base_name,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "correct_fields_count": TP,
        "wrong_fields_count": len(wrong_fields),
        "partial_matches_count": len(partial_fields),
        "missing_fields_count": len(missing_fields),
        "extra_fields_count": len(extra_fields),
    }
    return report, base_name


def main():
    parser = argparse.ArgumentParser(description="Validate Gemini extraction output against ground truth.")
    parser.add_argument(
        "--prediction-file",
        "-p",
        help="Specific prediction JSON file to evaluate. Defaults to all files in --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(OUTPUT_FOLDER),
        help="Directory containing prediction JSON files.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        "-g",
        default=str(GROUND_TRUTH_FOLDER),
        help="Directory containing ground-truth JSON files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    truth_dir = Path(args.ground_truth_dir).expanduser()

    if args.prediction_file:
        files = [Path(args.prediction_file).expanduser()]
    else:
        files = sorted(output_dir.glob("*.json"))

    if not files:
        print("No prediction files found.")
        return

    print("\n Running simplified validation (counts only)...\n")

    for out_file in files:
        result = evaluate_prediction(out_file, truth_dir)
        if not result:
            continue
        report, base_name = result
        save_path = SAVE_DIR / f"{base_name}_eval.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f" Saved clean report: {save_path}")

    print("\n DONE — simplified count-based evaluation JSON created!\n")


if __name__ == "__main__":
    main()
