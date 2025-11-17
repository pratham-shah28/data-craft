# model_1/v2_vertex/example_provider.py
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Any, Dict, List

# ------------------------------------------------------------
# Path setup
#
# We try to locate the repo root by walking up the directory tree
# and looking for a "data_pipeline" or "data-pipeline" folder.
# This makes it work both:
#   • locally (full repo layout)
#   • inside Docker (as long as the repo root is copied)
# ------------------------------------------------------------
HERE = Path(__file__).resolve()


def find_repo_root(start: Path) -> Path:
    """
    Walk upwards from `start` and look for a folder that contains
    either 'data_pipeline' or 'data-pipeline'. That parent is treated
    as the repo root. If nothing is found, fall back to start.parent.
    """
    for parent in [start] + list(start.parents):
        if (parent / "data_pipeline").exists() or (parent / "data-pipeline").exists():
            return parent
    # Fallback: just use the directory containing this file
    return start.parent


REPO_ROOT = find_repo_root(HERE)

# Support both spellings, prefer data_pipeline
DATA_PIPELINE = REPO_ROOT / "data_pipeline"
if not DATA_PIPELINE.exists():
    DATA_PIPELINE = REPO_ROOT / "data-pipeline"

# Default invoices directory (your current behavior)
INVOICES_DIR = DATA_PIPELINE / "data" / "unstructured" / "invoices"

SCRIPTS_DIR = DATA_PIPELINE / "scripts"
PDF2IMAGE_PATH = SCRIPTS_DIR / "pdf_2_image.py"

# ------------------------------------------------------------
# Import helper module from data-pipeline/scripts/pdf_2_image.py
# Expects: pdf_to_base64_images(pdf_path: str, output_json: bool)
# ------------------------------------------------------------
_spec = importlib.util.spec_from_file_location("dp_pdf2image_helper", str(PDF2IMAGE_PATH))
_dp_pdf2image = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_dp_pdf2image)  # type: ignore[attr-defined]

pdf_to_base64_images = _dp_pdf2image.pdf_to_base64_images  # function from your helper


def build_examples_manifest(
    invoices_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Scans `invoices_dir` (default: DATA_PIPELINE/data/unstructured/invoices) for pairs:
      invoice_X.pdf + invoice_X.json

    Uses pdf_to_base64_images(...) to convert PDF pages → list of:
      { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } }

    Returns the list of example dicts; JSONL writing is currently commented out.
    """
    invoices = invoices_dir or INVOICES_DIR
    manifest_path = manifest_path or (invoices / "examples.jsonl")

    records: List[Dict[str, Any]] = []

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
            "images": images,          # [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}, ...]
            "expected_json": expected,
        }
        records.append(rec)

    # If you want to re-enable JSONL writing, uncomment:
    # with manifest_path.open("w", encoding="utf-8") as mf:
    #     for rec in records:
    #         mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


if __name__ == "__main__":
    # Default: read from INVOICES_DIR and (optionally) write examples.jsonl there
    build_examples_manifest()
