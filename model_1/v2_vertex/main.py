# inference.py
import os, json, base64
from google.cloud import aiplatform as vertex_ai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from langchain_core.output_parsers import JsonOutputParser
from example_provider import build_examples_manifest
from data_pipeline.scripts.pdf_2_image import pdf_to_base64_images


# --- env config ---
PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID") or "YOUR_PROJECT_ID"
LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
MODEL    = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
TEMP     = float(os.environ.get("VERTEX_TEMPERATURE", "0.0"))
MAX_TOKS = int(os.environ.get("VERTEX_MAX_TOKENS", "2048"))
TARGET_PDF = os.environ.get("TARGET_PDF", "data-pipeline/unstructured/invoices/new_invoice.pdf")


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
vertex_ai.init(project=PROJECT, location=LOCATION)
model = GenerativeModel(MODEL)

# 1. Few-shot examples (from provider)
examples = build_examples_manifest()[:3]  # 3 exemplars

parts = [Part.from_text(
    "You are an information-extraction model. "
    "Study these examples (document images + their JSON). "
    "Then extract structured JSON for the new document."
)]

for i, ex in enumerate(examples, 1):
    parts.append(Part.from_text(f"Example {i}:"))
    for img in ex["images"]:
        parts.append(dataurl_to_part(img["image_url"]["url"]))
    parts.append(Part.from_text(json.dumps(ex["expected_json"], indent=2)))

# 2. Add new document for prediction
parts.append(Part.from_text("Now extract structured JSON for this new document:"))
images, _ = pdf_to_base64_images(TARGET_PDF, output_json=False)
for img in images:
    parts.append(dataurl_to_part(img["image_url"]["url"]))
parts.append(Part.from_text("Return ONLY valid JSON—no text outside JSON."))

# 3. Generate + parse
cfg = GenerationConfig(
    temperature=TEMP,
    max_output_tokens=MAX_TOKS,
    response_mime_type="application/json"
)
resp = model.generate_content(parts, generation_config=cfg)

print(json.dumps(safe_json_parse(resp.text or "{}"), indent=2, ensure_ascii=False))
