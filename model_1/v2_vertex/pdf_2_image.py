import base64
import io
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from pdf2image import convert_from_bytes


def encode_page(page):
    byte_io = io.BytesIO()
    page.save(byte_io, format="PNG")  
    base64_data = base64.b64encode(byte_io.getvalue()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_data}"
        }
    }


def pdf_to_base64_images(pdf_path: str, output_json: bool = True):
    pdf_path = Path(pdf_path).resolve()

    # Project structure: scripts/ and data/unstructured on same level
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir.parent / "data" / "unstructured"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{pdf_path.stem}_images.json"

    # Read PDF
    with open(pdf_path, "rb") as f:
        document_bytes = f.read()

    # Convert PDF pages to images and encode them
    pages = convert_from_bytes(document_bytes)
    with ThreadPoolExecutor() as executor:
        user_content = list(executor.map(encode_page, pages))

    output_data = {
        "pdf": str(pdf_path),
        "page_count": len(user_content),
        "images": user_content,
    }

    if output_json:
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved extracted images to: {output_file}")

    return user_content, output_data