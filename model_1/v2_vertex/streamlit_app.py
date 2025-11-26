# Streamlit front end for Gemini JSON extraction
import base64
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from example_provider import INVOICES_DIR
from inference import run_vertex_extraction
from typing import Dict

# Ensure pdf_to_base64_images is importable (same path trick as inference.py)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(ROOT, "data-pipeline", "scripts"))
from pdf_2_image import pdf_to_base64_images  # noqa: E402


def save_json(output_dir: Path, doc_type: str, payload: Dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{doc_type}_output_{timestamp}.json"
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_file


def main():
    st.set_page_config(page_title="Vertex AI Invoice Extraction", layout="wide")
    st.title("Vertex AI JSON Extraction")
    st.write("Upload a PDF, provide few-shot examples directory, and get structured JSON back.")

    try:
        # st.secrets raises if no secrets file exists; keep env/constant fallback
        sample_secret = st.secrets.get("SAMPLE_PATH", None)  # type: ignore[attr-defined]
    except Exception:
        sample_secret = None
    default_sample = Path(sample_secret or os.environ.get("SAMPLE_PATH", INVOICES_DIR))
    default_output = Path(__file__).resolve().parent / "output"

    with st.sidebar:
        st.header("Configuration")
        sample_dir = Path(st.text_input("Examples folder (PDF + JSON pairs)", value=str(default_sample)))
        doc_type = st.text_input("Document type", value=os.environ.get("DOC_TYPE", sample_dir.name or "invoices"))
        max_examples = st.number_input("Max examples", min_value=1, max_value=10, value=int(os.environ.get("VERTEX_EXAMPLE_COUNT", "3")))
        temperature = float(st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(os.environ.get("VERTEX_TEMPERATURE", "0.0")), step=0.1))
        max_toks = int(st.number_input("Max output tokens", min_value=512, max_value=8192, value=int(os.environ.get("VERTEX_MAX_TOKENS", "2048")), step=256))

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:
        # Persist upload to a temp file so it survives reruns
        if "uploaded_path" not in st.session_state:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(uploaded.getbuffer())
            tmp.flush()
            st.session_state.uploaded_path = tmp.name
        target_pdf = Path(st.session_state.uploaded_path)
        st.success(f"Loaded file: {uploaded.name} ({target_pdf})")
    else:
        target_pdf = None

    run_clicked = st.button("Run inference", type="primary")

    if run_clicked:
        if not target_pdf:
            st.error("Please upload a PDF first.")
            return
        if not sample_dir.exists():
            st.error(f"Sample directory not found: {sample_dir}")
            return

        with st.spinner("Calling AI..."):
            try:
                result = run_vertex_extraction(
                    target_pdf=target_pdf,
                    sample_path=sample_dir,
                    doc_type=doc_type.strip() or sample_dir.name,
                    max_examples=int(max_examples),
                    temperature=float(temperature),
                    max_toks=int(max_toks),
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Inference failed: {e}")
                return

        st.subheader("Model JSON")
        st.json(result)

        saved = save_json(default_output, doc_type.strip() or sample_dir.name, result)
        st.success(f"Saved to: {saved}")

        st.download_button(
            label="Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name=saved.name,
            mime="application/json",
        )

        # Provide a small preview of the PDF as base64 if single page
        try:
            imgs, _ = pdf_to_base64_images(str(target_pdf), output_json=False)
            if imgs:
                # Only show first page to avoid heavy UI
                first = imgs[0]["image_url"]["url"].split(",", 1)[1]
                st.image(base64.b64decode(first), caption="First page preview")
        except Exception:
            pass


if __name__ == "__main__":
    main()
