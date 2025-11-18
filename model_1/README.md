
# Dynamic Few-Shot Prompting for Document-Aware Extraction
This repository implements **dynamic few-shot prompting** for structured document understanding. The system automatically selects the most relevant example prompts based on the **document type** (invoices, reports, tax forms, certificates, contracts, etc.) and constructs an optimized prompt for a large language model (LLM). This enables consistent, accurate JSON extraction across highly heterogeneous business documents.

---

## 🔍 Overview
Typical few-shot prompting requires manually selecting examples, which fails when scaling to many document types.
**Dynamic few-shot prompting** solves this by:

1. Detecting the document’s type (via filename pattern, classifier output, template tag, or user metadata).
2. Fetching the best examples from a repository of JSONL prompt examples.
3. Building a prompt at inference-time using:
   - schema metadata
   - 2–5 nearest examples of the same category
   - instructions tailored to the document family
4. Calling the LLM (e.g., Gemini, Llama, GPT) using JSONOutputParser for deterministic structured output.

---

## 📁 Repository Structure
project/
│
├── data-pipeline/
│   ├── examples/
│   │   ├── invoices.jsonl
│   │   ├── statements.jsonl
│   │   ├── tax_forms.jsonl
│   │   ├── certificates.jsonl
│   │   └── contracts.jsonl
│   ├── scripts/
│   │   ├── build_examples.py
│   │   └── inference.py
│   └── schemas/
│       └── invoice_schema.json
│
├── model/
│   └── llm_client.py
│
└── README.md

---

## ⚙️ How It Works

### 1. Document Type Identification
Your inference script determines the document type by:
- reading metadata
- applying heuristics (e.g., “invoice_2024_*.pdf”)
- or running a light document-type classifier

### 2. Dynamic Few-Shot Retrieval
`build_examples.py` loads the correct JSONL file and selects:
- the closest examples based on similarity
- or a fixed number of curated exemplars for that category

### 3. Prompt Construction
The system creates a prompt containing:
- high-level instructions
- schema definition
- K example prompts
- the current document (image, OCR text, or both)

### 4. LLM Inference with JSONOutputParser
The model uses a strict parser to guarantee valid schema-compliant output.

---

## 🧩 Supported Document Families
- Invoices → totals, vendor metadata, tax lines
- Financial statements → balance sheet fields
- Tax forms → form identifiers and taxpayer metadata
- Certificates → issuer, recipient, validity dates
- Contracts → parties, obligations, signature metadata

---

## 🚀 Running Inference

Basic command:
python inference.py --input mydoc.pdf --doc_type invoice

Automatic document type detection:
python inference.py --input sample.pdf --auto

---

## 📦 Extending the System
To add support for new document types:

1. Create `examples/<type>.jsonl`
2. Create `<type>_schema.json`
3. Register the type in `build_examples.py`

No model retraining required.

---

## 🔒 Notes
- Compatible with any LLM (Gemini, GPT, Llama).
- Vision models supported when input is images instead of text.
- Designed for production pipelines on Vertex AI, AWS, Azure, or local.

---

## 💬 Support
Contact the maintainer for adding schemas, optimizing examples, or connecting outputs to downstream Knowledge Graph ingestion.


