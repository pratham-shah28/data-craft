import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "mlflow"))
sys.path.insert(0, str(ROOT / "data-pipeline" / "scripts"))
sys.path.insert(0, str(ROOT / "model_1" / "v2_vertex"))

import tracker
import config

MLflowTracker = tracker.MLflowTracker
EXPERIMENTS = config.EXPERIMENTS
MLFLOW_TRACKING_URI = config.MLFLOW_TRACKING_URI
DEFAULT_TAGS = config.DEFAULT_TAGS

from google.cloud import aiplatform as vertex_ai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from langchain_core.output_parsers import JsonOutputParser
from example_provider import build_examples_manifest
from pdf_2_image import pdf_to_base64_images
import base64


class LLM1Evaluator:
    
    def __init__(self, project_id, location, model_name, temperature, max_tokens):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        vertex_ai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
        
        self.tracker = MLflowTracker(
            experiment_name=EXPERIMENTS["llm1_document_extraction"],
            tracking_uri=MLFLOW_TRACKING_URI
        )
        
    def dataurl_to_part(self, url: str) -> Part:
        raw = base64.b64decode(url.split(",", 1)[1])
        return Part.from_data(raw, mime_type="image/png")
    
    def safe_json_parse(self, text: str):
        try:
            return JsonOutputParser().parse(text)
        except Exception:
            i, j = text.find("{"), text.rfind("}")
            return json.loads(text[i:j+1]) if i != -1 and j != -1 and j > i else {}
    
    def extract_from_document(self, pdf_path, examples):
        parts = [Part.from_text(
            "You are an information extraction model. "
            "Study the following examples (document images + JSON). "
            "Then extract all information structured JSON for the new document below."
        )]
        
        for i, ex in enumerate(examples, 1):
            parts.append(Part.from_text(f"Example {i}:"))
            for img in ex["images"]:
                parts.append(self.dataurl_to_part(img["image_url"]["url"]))
            parts.append(Part.from_text(json.dumps(ex["expected_json"], indent=2)))
        
        parts.append(Part.from_text("Now extract structured JSON for this new document:"))
        images, _ = pdf_to_base64_images(str(pdf_path), output_json=False)
        for img in images:
            parts.append(self.dataurl_to_part(img["image_url"]["url"]))
        parts.append(Part.from_text("Return ONLY valid JSON — no explanation or extra text."))
        
        resp = self.model.generate_content(
            parts,
            generation_config=GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json"
            ),
        )
        
        return self.safe_json_parse(resp.text or "{}")
    
    def calculate_field_accuracy(self, predicted, ground_truth):
        if not predicted or not ground_truth:
            return 0.0
        
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        pred_flat = flatten_dict(predicted)
        gt_flat = flatten_dict(ground_truth)
        
        all_keys = set(pred_flat.keys()) | set(gt_flat.keys())
        if not all_keys:
            return 0.0
        
        correct = 0
        for key in all_keys:
            pred_val = pred_flat.get(key)
            gt_val = gt_flat.get(key)
            
            if pred_val == gt_val:
                correct += 1
        
        return correct / len(all_keys)
    
    def evaluate_document_type(self, doc_type, data_dir):
        doc_dir = Path(data_dir) / doc_type
        
        if not doc_dir.exists():
            print(f"Warning: {doc_dir} does not exist")
            return None
        
        pdf_files = list(doc_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"Warning: No PDF files in {doc_dir}")
            return None
        
        examples = build_examples_manifest()
        
        results = []
        
        for pdf_path in pdf_files:
            json_path = pdf_path.with_suffix('.json')
            
            if not json_path.exists():
                continue
            
            print(f"Processing: {pdf_path.name}")
            
            start_time = time.time()
            
            try:
                prediction = self.extract_from_document(pdf_path, examples)
                
                with open(json_path, 'r') as f:
                    ground_truth_data = json.load(f)
                
                if "0_expected" in ground_truth_data:
                    ground_truth = ground_truth_data["0_expected"]
                else:
                    ground_truth = ground_truth_data
                
                accuracy = self.calculate_field_accuracy(prediction, ground_truth)
                
                extraction_time = time.time() - start_time
                
                results.append({
                    "document": pdf_path.name,
                    "accuracy": accuracy,
                    "extraction_time": extraction_time,
                    "success": True
                })
                
                print(f"  Accuracy: {accuracy:.2%}, Time: {extraction_time:.2f}s")
                
            except Exception as e:
                extraction_time = time.time() - start_time
                results.append({
                    "document": pdf_path.name,
                    "accuracy": 0.0,
                    "extraction_time": extraction_time,
                    "success": False,
                    "error": str(e)
                })
                print(f"  Error: {str(e)}")
        
        return results
    
    def run_evaluation(self, data_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"llm1_eval_{timestamp}"
        
        tags = {
            **DEFAULT_TAGS,
            "model": self.model_name,
            "evaluation_type": "document_extraction"
        }
        
        self.tracker.start_run(run_name=run_name, tags=tags)
        
        params = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "project_id": self.project_id,
            "location": self.location
        }
        
        self.tracker.log_params(params)
        
        print("\n" + "=" * 60)
        print("LLM1 EVALUATION: Document Extraction")
        print("=" * 60 + "\n")
        
        document_types = ["invoices", "vehicle_insurance", "us_tax"]
        all_results = {}
        
        for doc_type in document_types:
            print(f"\nEvaluating: {doc_type}")
            print("-" * 40)
            
            results = self.evaluate_document_type(doc_type, data_dir)
            
            if results:
                all_results[doc_type] = results
                
                successful = [r for r in results if r["success"]]
                
                if successful:
                    avg_accuracy = sum(r["accuracy"] for r in successful) / len(successful)
                    avg_time = sum(r["extraction_time"] for r in successful) / len(successful)
                    
                    metrics = {
                        f"{doc_type}_avg_accuracy": avg_accuracy,
                        f"{doc_type}_avg_time": avg_time,
                        f"{doc_type}_total_docs": len(results),
                        f"{doc_type}_successful": len(successful),
                        f"{doc_type}_failed": len(results) - len(successful)
                    }
                    
                    self.tracker.log_metrics(metrics)
                    
                    print(f"\nResults for {doc_type}:")
                    print(f"  Average Accuracy: {avg_accuracy:.2%}")
                    print(f"  Average Time: {avg_time:.2f}s")
                    print(f"  Success Rate: {len(successful)}/{len(results)}")
        
        if all_results:
            all_successful = []
            for doc_results in all_results.values():
                all_successful.extend([r for r in doc_results if r["success"]])
            
            if all_successful:
                overall_accuracy = sum(r["accuracy"] for r in all_successful) / len(all_successful)
                overall_time = sum(r["extraction_time"] for r in all_successful) / len(all_successful)
                
                overall_metrics = {
                    "overall_avg_accuracy": overall_accuracy,
                    "overall_avg_time": overall_time,
                    "overall_total_docs": len(all_successful),
                    "overall_document_types": len(all_results)
                }
                
                self.tracker.log_metrics(overall_metrics)
                
                print("\n" + "=" * 60)
                print(f"Overall Accuracy: {overall_accuracy:.2%}")
                print(f"Overall Avg Time: {overall_time:.2f}s")
                print(f"Total Documents: {len(all_successful)}")
                print("=" * 60 + "\n")
        
        self.tracker.log_dict(all_results, "evaluation_results.json")
        
        self.tracker.end_run(status="FINISHED")
        
        return all_results


if __name__ == "__main__":
    PROJECT_ID = "datacraft-data-pipeline"
    LOCATION = "us-central1"
    MODEL_NAME = "gemini-2.5-flash"
    TEMPERATURE = 0.0
    MAX_TOKENS = 8192
    
    DATA_DIR = ROOT / "data-pipeline" / "data" / "unstructured"
    
    evaluator = LLM1Evaluator(
        project_id=PROJECT_ID,
        location=LOCATION,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    results = evaluator.run_evaluation(DATA_DIR)
    
    print("\nEvaluation complete. Check MLflow UI for detailed results.")
