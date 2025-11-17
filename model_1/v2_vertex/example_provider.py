# model_1/v2_vertex/example_provider.py
from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

# ------------------------------------------------------------
# Path setup (repo root is two levels up from this file)
#   repo_root/
#     ├─ data-pipeline/
#     └─ model_1/
#         └─ v2_vertex/
#             └─ example_provider.py  (this file)
# ------------------------------------------------------------
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
DATA_PIPELINE = REPO_ROOT / "data_pipeline"

# Prefer the correct folder; fallback to common misspelling
INVOICES_DIR = DATA_PIPELINE /"data"/ "unstructured" / "invoices"


SCRIPTS_DIR = DATA_PIPELINE / "scripts"
PDF2IMAGE_PATH = SCRIPTS_DIR / "pdf_2_image.py"


# ------------------------------------------------------------
# Import helper module from data-pipeline/scripts/pdf2image.py
# (avoids clashing with the 'pdf2image' PyPI package name)
# Expects the module to expose: pdf_to_base64_images(pdf_path: str, output_json: bool)
# ------------------------------------------------------------
_spec = importlib.util.spec_from_file_location("dp_pdf2image_helper", str(PDF2IMAGE_PATH))
_dp_pdf2image = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_dp_pdf2image)  # type: ignore[attr-defined]

pdf_to_base64_images = _dp_pdf2image.pdf_to_base64_images  # function from your helper


def build_examples_manifest(manifest_path: Path | None = None) -> List[Dict[str, Any]]:
    """
    Scans data-pipeline/(unstructured|unstrcutured)/invoices for pairs:
      invoice_X.pdf + invoice_X.json
    Uses pdf_to_base64_images(...) to convert PDF pages → {type:'image_url', image_url:{url:'data:image/png;base64,...'}}
    Writes a JSONL manifest (examples.jsonl) and returns the list of example dicts.
    """
    invoices = INVOICES_DIR
    manifest_path = manifest_path or (invoices / "examples.jsonl")

    records: List[Dict[str, Any]] = []
    written = 0

    # Iterate PDFs in sorted order; expect same-stem .json
    for pdf_path in sorted(invoices.glob("*.pdf")):
        stem = pdf_path.stem
        json_path = invoices / f"{stem}.json"
        if not json_path.exists():
            # Skip if there is no expected JSON for the PDF
            continue

        # 1) Convert PDF → base64 data-URL images (no sidecar image.json files)
        images, meta = pdf_to_base64_images(str(pdf_path), output_json=False)

        # 2) Load expected JSON (any shape)
        try:
            expected = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            expected = {"_raw": json_path.read_text(encoding="utf-8", errors="ignore")}

        rec = {
            "id": stem,
            "page_count": meta.get("page_count", len(images)),
            "images": images,              # list of {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
            "expected_json": expected
        }
        records.append(rec)

    # # 3) Write JSONL manifest
    # with (manifest_path).open("w", encoding="utf-8") as mf:
    #     for rec in records:
    #         mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    #         written += 1
    return records


if __name__ == "__main__":
    build_examples_manifest()  # writes examples.jsonl next to the invoices
