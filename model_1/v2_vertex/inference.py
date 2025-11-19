# inference.py
import argparse
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from google.cloud import aiplatform as vertex_ai
from langchain_core.output_parsers import JsonOutputParser
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part

from example_provider import INVOICES_DIR, build_examples_manifest

# --- import pdf_to_base64_images from sibling folder ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(ROOT, "data-pipeline", "scripts"))
from pdf_2_image import pdf_to_base64_images  # noqa: E402


def dataurl_to_part(url: str) -> Part:
    """Convert base64 data URL → Gemini Part"""
    raw = base64.b64decode(url.split(",", 1)[1])
    return Part.from_data(raw, mime_type="image/png")


def safe_json_parse(text: str):
    """Parse model text safely into JSON"""
    try:
        return JsonOutputParser().parse(text)
    except Exception:
        i, j = text.find("{"), text.rfind("}")
        return json.loads(text[i:j + 1]) if i != -1 and j != -1 and j > i else {}


def main():
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID") or "mlops-472423"
    location = os.environ.get("VERTEX_LOCATION", "us-central1")
    model_name = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
    temperature = float(os.environ.get("VERTEX_TEMPERATURE", "0.0"))
    max_toks = int(os.environ.get("VERTEX_MAX_TOKENS", "8192"))

    default_sample = Path(os.environ.get("SAMPLE_PATH") or INVOICES_DIR)
    default_target = os.environ.get("TARGET_PDF") or str(default_sample / "invoice_6.pdf")

    parser = argparse.ArgumentParser(description="Run Gemini JSON extraction on a PDF.")
    parser.add_argument(
        "--sample-path",
        "-s",
        default=str(default_sample),
        help="Directory containing (PDF, JSON) pairs for few-shot examples.",
    )
    parser.add_argument(
        "--doc-type",
        "-d",
        default=os.environ.get("DOC_TYPE", "invoices"),
        help="Logical name for the document type (stored in output JSON).",
    )
    parser.add_argument(
        "--target-path",
        "-t",
        default=default_target,
        help="PDF file to extract structured JSON from.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(Path(__file__).resolve().parent / "output"),
        help="Directory to store model predictions.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=int(os.environ.get("VERTEX_EXAMPLE_COUNT", "3")),
        help="Number of few-shot exemplars to include.",
    )
    parser.add_argument(
        "--path-only",
        action="store_true",
        help="Suppress verbose output and print only the saved JSON path (useful for Airflow).",
    )
    args = parser.parse_args()

    sample_path = Path(args.sample_path).expanduser()
    target_path = Path(args.target_path or sample_path / "invoice_6.pdf").expanduser()
    doc_type = (args.doc_type or sample_path.name).strip() or sample_path.name
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sample_path.exists():
        raise FileNotFoundError(f"Sample path not found: {sample_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target PDF not found: {target_path}")

    vertex_ai.init(project=project, location=location)
    model = GenerativeModel(model_name)

    examples = build_examples_manifest(invoices_dir=sample_path)
    if not examples:
        raise RuntimeError(f"No example (PDF, JSON) pairs found under {sample_path}")
    examples = examples[: max(1, args.max_examples)]

    instructions = (
        "You are an information extraction model. "
        "Study the following examples (document images + JSON). "
        "Then extract all information structured JSON for the new document below."
    )
    parts = [Part.from_text(instructions)]

    for i, ex in enumerate(examples, 1):
        parts.append(Part.from_text(f"Example {i}:"))
        for img in ex["images"]:
            parts.append(dataurl_to_part(img["image_url"]["url"]))
        parts.append(Part.from_text(json.dumps(ex["expected_json"], indent=2)))

    parts.append(Part.from_text("Now extract structured JSON for this new document:"))
    images, _ = pdf_to_base64_images(str(target_path), output_json=False)
    for img in images:
        parts.append(dataurl_to_part(img["image_url"]["url"]))
    parts.append(Part.from_text("Return ONLY valid JSON — no explanation or extra text."))

    resp = model.generate_content(
        parts,
        generation_config=GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_toks,
            response_mime_type="application/json",
        ),
    )

    parsed = safe_json_parse(resp.text or "{}")
    parsed["document_type"] = doc_type
    parsed["origin_file"] = str(target_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{doc_type}_output_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    if args.path_only:
        print(str(output_file))
    else:
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        print(f"\n✅ JSON saved to: {output_file}\n")


if __name__ == "__main__":
    main()
