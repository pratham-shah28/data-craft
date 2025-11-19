import os
import json
import math
from pathlib import Path
from Levenshtein import ratio as similarity


OUTPUT_FOLDER = Path(__file__).resolve().parent / "output"

GROUND_TRUTH_FOLDER = Path(__file__).resolve().parents[2] / "data-pipeline" / "data" / "unstructured" / "invoices"

SAVE_DIR = Path(__file__).resolve().parent / "eval_simple"
os.makedirs(SAVE_DIR, exist_ok=True)



def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_filename(origin):
    base = os.path.basename(origin)
    return base.replace(".pdf", "")


def flatten_json(d, parent="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def simple_match(a, b):
    if a == b:
        return True

    # numeric tolerance for floats
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0.01)

    # string similarity
    if isinstance(a, str) and isinstance(b, str):
        t, p = a.strip().lower(), b.strip().lower()
        if t == p:
            return True
        if similarity(t, p) > 0.80:
            return "partial"

    return False



output_files = list(Path(OUTPUT_FOLDER).glob("*.json"))
if not output_files:
    print("No prediction files found.")
    exit()

print("\n Running simplified validation (counts only)...\n")

for out_file in output_files:
    pred_json = load_json(out_file)

    origin_file = pred_json.get("origin_file")
    if not origin_file:
        print(f"⚠ Skipping {out_file} (no origin_file)")
        continue

    base_name = extract_filename(origin_file)
    truth_path = Path(GROUND_TRUTH_FOLDER) / f"{base_name}.json"

    if not truth_path.exists():
        print(f"⚠ No ground truth for {base_name}. Skipping.")
        continue

    truth_json = load_json(truth_path)["0_expected"]

    # flatten both
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

    # model produced additional fields not in truth
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

        # simple counts
        "correct_fields_count": TP,
        "wrong_fields_count": len(wrong_fields),
        "partial_matches_count": len(partial_fields),
        "missing_fields_count": len(missing_fields),
        "extra_fields_count": len(extra_fields),

        
    }

    save_path = Path(SAVE_DIR) / f"{base_name}_eval.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f" Saved clean report: {save_path}")

print("\n DONE — simplified count-based evaluation JSON created!\n")
