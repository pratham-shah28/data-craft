"""
Unstructured Data Handler - Process PDFs/images and store in BigQuery
"""

import os
import json
import base64
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
import logging

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'data-pipeline' / 'scripts'))
from pdf_2_image import pdf_to_base64_images

# Add model_1 path
sys.path.insert(0, str(Path(__file__).parent.parent / 'model_1' / 'v2_vertex'))
from example_provider import build_examples_manifest


class UnstructuredDataHandler:
    """
    Handle unstructured data (PDFs, images) extraction and BigQuery storage
    """
    
    def __init__(self, project_id: str, dataset_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.region = region
        
        # Initialize clients
        self.bq_client = bigquery.Client(project=project_id)
        vertexai.init(project=project_id, location=region)
        
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_name = "gemini-2.0-flash-exp"
        self.model = GenerativeModel(self.model_name)
    
    def dataurl_to_part(self, url: str) -> Part:
        """Convert base64 data URL to Gemini Part"""
        raw = base64.b64decode(url.split(",", 1)[1])
        return Part.from_data(raw, mime_type="image/png")
    
    def safe_json_parse(self, text: str) -> Dict:
        """Parse model text safely into JSON"""
        try:
            # Try direct JSON parse
            return json.loads(text)
        except Exception:
            # Extract JSON from text
            i, j = text.find("{"), text.rfind("}")
            if i != -1 and j != -1 and j > i:
                return json.loads(text[i:j+1])
            return {}
    
    def extract_from_pdf(
        self, 
        pdf_path: str, 
        doc_type: str,
        examples_dir: str = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from PDF using few-shot learning
        
        Args:
            pdf_path: Path to PDF file
            doc_type: Type of document (invoices, insurance, etc.)
            examples_dir: Directory containing example PDFs and JSONs
            
        Returns:
            Extracted JSON data
        """
        self.logger.info(f"Extracting data from: {pdf_path}")
        
        # Build few-shot examples
        examples = []
        if examples_dir and os.path.exists(examples_dir):
            try:
                examples = build_examples_manifest(manifest_path=examples_dir)
                examples = examples[:3]  # Use 3 examples
                self.logger.info(f"Loaded {len(examples)} examples from {examples_dir}")
            except Exception as e:
                self.logger.warning(f"Could not load examples: {str(e)}")
                examples = []
        
        # Build prompt parts
        parts = [Part.from_text(
            "You are an information extraction model. "
            "Study the following examples (document images + JSON). "
            "Then extract all information as structured JSON for the new document below."
        )]
        
        # Add examples
        for i, ex in enumerate(examples, 1):
            parts.append(Part.from_text(f"Example {i}:"))
            
            # Handle images - check if it's the correct format
            if 'images' in ex and isinstance(ex['images'], list):
                for img in ex['images']:
                    # Check if img is dict with correct structure
                    if isinstance(img, dict):
                        if 'image_url' in img and isinstance(img['image_url'], dict) and 'url' in img['image_url']:
                            # Correct format: {"type": "image_url", "image_url": {"url": "data:..."}}
                            parts.append(self.dataurl_to_part(img['image_url']['url']))
                        elif 'url' in img:
                            # Alternative format: {"url": "data:..."}
                            parts.append(self.dataurl_to_part(img['url']))
                    elif isinstance(img, str):
                        # Direct string format
                        parts.append(self.dataurl_to_part(img))
            
            # Add expected JSON
            if 'expected_json' in ex:
                parts.append(Part.from_text(json.dumps(ex['expected_json'], indent=2)))
        
        # Add target PDF
        parts.append(Part.from_text("Now extract structured JSON for this new document:"))
        
        try:
            images, meta = pdf_to_base64_images(pdf_path, output_json=False)
        except Exception as e:
            self.logger.error(f"Failed to convert PDF to images: {str(e)}")
            raise
        
        # Add target document images
        if isinstance(images, list):
            for img in images:
                if isinstance(img, dict):
                    if 'image_url' in img and isinstance(img['image_url'], dict) and 'url' in img['image_url']:
                        parts.append(self.dataurl_to_part(img['image_url']['url']))
                    elif 'url' in img:
                        parts.append(self.dataurl_to_part(img['url']))
                elif isinstance(img, str):
                    parts.append(self.dataurl_to_part(img))
        
        parts.append(Part.from_text("Return ONLY valid JSON — no explanation or extra text."))
        
        # Generate response
        try:
            response = self.model.generate_content(
                parts,
                generation_config=GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=8192,
                    response_mime_type="application/json"
                )
            )
            
            extracted_data = self.safe_json_parse(response.text or "{}")
            
            # Add metadata
            extracted_data['_metadata'] = {
                'document_type': doc_type,
                'origin_file': os.path.basename(pdf_path),
                'extracted_at': datetime.now().isoformat(),
                'page_count': meta.get('page_count', len(images)) if isinstance(meta, dict) else len(images)
            }
            
            self.logger.info(f"Successfully extracted data from {pdf_path}")
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}")
            raise
    
    def process_multiple_files(
        self,
        file_paths: List[str],
        doc_type: str,
        examples_dir: str = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PDF files and extract data from all
        
        Args:
            file_paths: List of PDF file paths
            doc_type: Document type
            examples_dir: Directory with examples
            
        Returns:
            List of extracted JSON objects
        """
        results = []
        
        for file_path in file_paths:
            try:
                extracted = self.extract_from_pdf(file_path, doc_type, examples_dir)
                results.append(extracted)
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {str(e)}")
                # Add error record
                results.append({
                    '_metadata': {
                        'document_type': doc_type,
                        'origin_file': os.path.basename(file_path),
                        'extracted_at': datetime.now().isoformat(),
                        'error': str(e)
                    }
                })
        
        return results
    
    def json_to_bigquery_schema(self, sample_json: Dict) -> List[bigquery.SchemaField]:
        """
        Infer BigQuery schema from sample JSON
        
        Args:
            sample_json: Sample JSON object
            
        Returns:
            List of BigQuery schema fields
        """
        schema = []
        
        def infer_type(value):
            if isinstance(value, bool):
                return "BOOLEAN"
            elif isinstance(value, int):
                return "INTEGER"
            elif isinstance(value, float):
                return "FLOAT"
            elif isinstance(value, str):
                return "STRING"
            elif isinstance(value, dict):
                return "JSON"
            elif isinstance(value, list):
                return "JSON"  # Store arrays as JSON
            else:
                return "STRING"
        
        for key, value in sample_json.items():
            field_type = infer_type(value)
            mode = "NULLABLE"
            
            schema.append(bigquery.SchemaField(
                name=key.replace(' ', '_').replace('-', '_').lower(),
                field_type=field_type,
                mode=mode
            ))
        
        return schema
    
    def store_in_bigquery(
        self,
        extracted_data: List[Dict[str, Any]],
        doc_type: str,
        table_suffix: str = "_processed"
    ) -> str:
        """
        Store extracted JSON data in BigQuery
        
        Args:
            extracted_data: List of extracted JSON objects
            doc_type: Document type (becomes table name)
            table_suffix: Suffix for table name
            
        Returns:
            Full table ID
        """
        if not extracted_data:
            raise ValueError("No data to store")
        
        # Create table name
        table_name = f"{doc_type.lower().replace(' ', '_').replace('-', '_')}{table_suffix}"
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        self.logger.info(f"Storing data in BigQuery table: {table_id}")
        
        # Flatten JSON for DataFrame
        flattened_data = []
        for record in extracted_data:
            flat_record = {}
            
            # Flatten nested structures
            for key, value in record.items():
                if isinstance(value, dict):
                    # Store dict as JSON string
                    flat_record[key] = json.dumps(value)
                elif isinstance(value, list):
                    # Store list as JSON string
                    flat_record[key] = json.dumps(value)
                else:
                    flat_record[key] = value
            
            flattened_data.append(flat_record)
        
        # Create DataFrame
        df = pd.DataFrame(flattened_data)
        
        # Normalize column names
        df.columns = [
            col.replace(' ', '_').replace('-', '_').lower()
            for col in df.columns
        ]
        
        # Configure load job - Use CSV format for dataframe
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",  # Append to existing table
            autodetect=True,  # Auto-detect schema
            source_format=bigquery.SourceFormat.CSV  # ✅ Use CSV format
        )
        
        # Load to BigQuery
        try:
            job = self.bq_client.load_table_from_dataframe(
                df,
                table_id,
                job_config=job_config
            )
            job.result()  # Wait for completion
            
            self.logger.info(f"✓ Loaded {len(df)} rows to {table_id}")
            
            return table_id
            
        except Exception as e:
            self.logger.error(f"Failed to load data to BigQuery: {str(e)}")
            raise
    
    def get_table_info(self, doc_type: str, table_suffix: str = "_processed") -> Dict:
        """
        Get information about a BigQuery table
        
        Args:
            doc_type: Document type
            table_suffix: Table suffix
            
        Returns:
            Table information
        """
        table_name = f"{doc_type.lower().replace(' ', '_').replace('-', '_')}{table_suffix}"
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        try:
            table = self.bq_client.get_table(table_id)
            
            return {
                'table_id': table_id,
                'num_rows': table.num_rows,
                'num_columns': len(table.schema),
                'created': table.created.isoformat() if table.created else None,
                'modified': table.modified.isoformat() if table.modified else None,
                'size_bytes': table.num_bytes
            }
        except Exception as e:
            self.logger.warning(f"Table {table_id} not found: {str(e)}")
            return None
    
    def list_unstructured_tables(self) -> List[Dict]:
        """
        List all unstructured data tables in the dataset
        
        Returns:
            List of table information
        """
        tables = []
        
        try:
            dataset_ref = self.bq_client.dataset(self.dataset_id)
            table_list = self.bq_client.list_tables(dataset_ref)
            
            for table in table_list:
                table_name = table.table_id
                
                # Only include _processed tables from unstructured data
                if table_name.endswith('_processed'):
                    full_table = self.bq_client.get_table(table.reference)
                    
                    # Check if table has _metadata column (indicator of unstructured data)
                    has_metadata = any(field.name == '_metadata' for field in full_table.schema)
                    
                    if has_metadata:
                        tables.append({
                            'name': table_name.replace('_processed', ''),
                            'table_id': f"{self.project_id}.{self.dataset_id}.{table_name}",
                            'rows': full_table.num_rows,
                            'columns': len(full_table.schema),
                            'size_mb': round(full_table.num_bytes / (1024 * 1024), 2),
                            'updated': full_table.modified.isoformat() if full_table.modified else None
                        })
            
            return tables
            
        except Exception as e:
            self.logger.error(f"Failed to list tables: {str(e)}")
            return []


if __name__ == "__main__":
    # Test the handler
    handler = UnstructuredDataHandler(
        project_id="datacraft-data-pipeline",
        dataset_id="datacraft_ml"
    )
    
    # Example usage
    pdf_path = "path/to/invoice.pdf"
    doc_type = "invoices"
    examples_dir = "path/to/examples"
    
    # Extract from single file
    extracted = handler.extract_from_pdf(pdf_path, doc_type, examples_dir)
    
    # Store in BigQuery
    table_id = handler.store_in_bigquery([extracted], doc_type)
    
    print(f"Data stored in: {table_id}")