# inference.py
import os, json, base64, sys
from datetime import datetime
from google.cloud import aiplatform as vertex_ai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from langchain_core.output_parsers import JsonOutputParser
import argparse
from example_provider import build_examples_manifest
from pathlib import Path

# --- import pdf_to_base64_images from sibling folder ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(ROOT, "data-pipeline", "scripts"))
from pdf_2_image import pdf_to_base64_images  
# --- env config ---



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
        return json.loads(text[i:j+1]) if i != -1 and j != -1 and j > i else {}


# --- main ---
def main():
    PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID") or "mlops-472423"
    LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
    MODEL    = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
    TEMP     = float(os.environ.get("VERTEX_TEMPERATURE", "0.0"))
    MAX_TOKS = int(os.environ.get("VERTEX_MAX_TOKENS", "8192"))
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    #TARGET_PDF = os.environ.get("TARGET_PDF", "mlops-project/data_pipeline/data/unstructured/vehicle_insurance/policy_1.pdf")

    parser = argparse.ArgumentParser(description="Run Gemini JSON extraction on a PDF.")
    parser.add_argument(
        "--sample-path", "-s",
        default="/Users/prathamshah/Desktop/projects/mlops project/mlops-project/data-pipeline/data/unstructured/invoices",
        help="Document type (default: invoices)"
    )
    parser.add_argument(
        "--doc-type", "-d",
        default="invoices",
        help="Document type (default: invoices)"
    )

    parser.add_argument(
        "--target-path", "-t",
        default="/Users/prathamshah/Desktop/projects/mlops project/mlops-project/data-pipeline/data/unstructured/invoices/invoice_6.pdf",
        help="Document type (default: invoices)"
    )
    args = parser.parse_args()
    sample_path = args.sample_path
    target_path = args.target_path
    doc_type = args.doc_type
    vertex_ai.init(project=PROJECT, location=LOCATION)
    model = GenerativeModel(MODEL)

    examples = build_examples_manifest(manifest_path=sample_path)
    examples = examples[:3]  # 3 exemplars

    parts = [Part.from_text(
        "You are an information extraction model. "
        "Study the following examples (document images + JSON). "
        "Then extract all information structured JSON for the new document below."
    )]

    for i, ex in enumerate(examples, 1):
        parts.append(Part.from_text(f"Example {i}:"))
        for img in ex["images"]:
            parts.append(dataurl_to_part(img["image_url"]["url"]))
        parts.append(Part.from_text(json.dumps(ex["expected_json"], indent=2)))

    # Add new PDF for prediction
    parts.append(Part.from_text("Now extract structured JSON for this new document:"))
    images, _ = pdf_to_base64_images(target_path, output_json=False)
    for img in images:
        parts.append(dataurl_to_part(img["image_url"]["url"]))
    parts.append(Part.from_text("Return ONLY valid JSON — no explanation or extra text."))

    # Generate + parse output
    resp = model.generate_content(
        parts,
        generation_config=GenerationConfig(
            temperature=TEMP,
            max_output_tokens=MAX_TOKS,
            response_mime_type="application/json"
        ),
    )
    print(json.dumps(safe_json_parse(resp.text or "{}"), indent=2, ensure_ascii=False))
    parsed = safe_json_parse(resp.text or "{}")
    parsed['document_type'] = doc_type  
    parsed['origin_file'] = target_path



    # Save output JSON to file
    ROOT_DIR = Path(__file__).resolve().parent
    output_dir = Path(ROOT_DIR) / "output"   
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{doc_type}_output_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    print(f"\n✅ JSON saved to: {output_file}\n")

if __name__ == "__main__":
    main()
