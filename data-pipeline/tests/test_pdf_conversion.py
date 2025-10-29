# test/test_pdf_conversion.py
import sys
from pathlib import Path

# Add data-pipeline root so "scripts" is importable
ROOT = Path(__file__).resolve().parents[1]   
sys.path.insert(0, str(ROOT))

from scripts.pdf_2_image import pdf_to_base64_images 

def test_pdf_conversion():
    pdf_path = Path(__file__).parent / "assets" / "test.pdf"

    images, metadata = pdf_to_base64_images(str(pdf_path), output_json=True)

    # ✅ Validate metadata fields
    assert metadata["page_count"] > 0
    assert isinstance(images, list)

    # ✅ Validate image_url structure
    first_img = images[0]
    assert "type" in first_img
    assert first_img["type"] == "image_url"
    assert "image_url" in first_img
    assert first_img["image_url"]["url"].startswith("data:image/png;base64,")

    print("✅ PDF to base64 conversion test passed!")

if __name__ == "__main__":
    test_pdf_conversion()
