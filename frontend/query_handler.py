# backend/query_handler.py
"""
Query Handler - Processes user queries using trained model
Integrates with model training pipeline components
"""

import sys
from pathlib import Path
import time

# Add model training scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'model-training' / 'scripts'))

from metadata_manager import MetadataManager
from prompts import build_prompt, FEW_SHOT_EXAMPLES
from query_executor import QueryExecutor

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import json
import re


class QueryHandler:
    """
    Handle user queries end-to-end
    
    Flow:
    1. Get dataset metadata
    2. Build prompt with context
    3. Call Gemini model
    4. Parse response
    5. Execute SQL on BigQuery
    6. Validate results
    7. Return complete answer with visualization
    """
    
    def __init__(self, config: dict):
        """
        Initialize query handler
        
        Args:
            config: Dictionary with:
                - project_id: GCP project ID
                - dataset_id: BigQuery dataset ID
                - bucket_name: GCS bucket name
                - region: GCP region
                - model_name: Best model name
                - table_name: Table to query
        """
        self.project_id = config['project_id']
        self.dataset_id = config['dataset_id']
        self.bucket_name = config['bucket_name']
        self.region = config['region']
        self.model_name = config['model_name']
        self.table_name = config['table_name']
        
        # Initialize components
        self.metadata_manager = MetadataManager(self.project_id, self.dataset_id)
        self.query_executor = QueryExecutor(self.project_id, self.dataset_id)
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.region)
        
        # Load best model configuration (from pipeline)
        self.generation_config = self._load_best_model_config()
    
    def _load_best_model_config(self) -> GenerationConfig:
        """Load best generation config from pipeline results"""
        try:
            # Try to load from pipeline outputs
            metadata_path = Path('/opt/airflow/outputs/best-model-responses')
            
            # Find most recent run
            if metadata_path.exists():
                runs = sorted(metadata_path.glob('*/best_model_metadata.json'))
                if runs:
                    with open(runs[-1]) as f:
                        metadata = json.load(f)
                        self.model_name = metadata.get('selected_model', self.model_name)
            
            # Default configuration (or from hyperparameter tuning)
            return GenerationConfig(
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                max_output_tokens=2048
            )
        except:
            # Fallback to defaults
            return GenerationConfig(
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                max_output_tokens=2048
            )
    
    def get_dataset_metadata(self, dataset_name: str) -> dict:
        """Get metadata for dataset"""
        try:
            metadata = self.metadata_manager.get_metadata(dataset_name)
            return metadata
        except:
            return {}
    
    def process_query(self, user_query: str, dataset_name: str) -> dict:
        """
        Process user query end-to-end
        
        Args:
            user_query: Natural language question
            dataset_name: Dataset to query
            
        Returns:
            Dictionary with:
                - status: 'success' or 'error'
                - sql_query: Generated SQL
                - visualization: Viz config
                - explanation: Model explanation
                - visualization_data: Actual data (DataFrame)
                - natural_language_answer: Human-readable answer
                - execution_time: Total time taken
                - validation_checks: Validation results
        """
        start_time = time.time()
        
        result = {
            'status': 'processing',
            'user_query': user_query,
            'dataset_name': dataset_name
        }
        
        try:
            # Step 1: Get dataset metadata
            metadata_result = self.metadata_manager.get_metadata(dataset_name)
            
            if not metadata_result:
                raise ValueError(f"No metadata found for dataset: {dataset_name}")
            
            llm_context = metadata_result['llm_context']
            
            # Step 2: Build prompt
            prompt = build_prompt(user_query, llm_context, FEW_SHOT_EXAMPLES)
            
            # Step 3: Call Gemini model
            model = GenerativeModel(self.model_name, generation_config=self.generation_config)
            response = model.generate_content(prompt)
            response_text = response.text
            
            # Step 4: Parse response
            parsed = self._parse_response(response_text)
            
            result['sql_query'] = parsed['sql_query']
            result['visualization'] = parsed['visualization']
            result['explanation'] = parsed['explanation']
            
            # Step 5: Execute SQL on BigQuery
            execution_result = self.query_executor.execute_and_validate(
                user_query=user_query,
                sql_query=parsed['sql_query'],
                table_name=self.table_name,
                visualization=parsed['visualization']
            )
            
            # Step 6: Extract results
            if execution_result['execution_status'] == 'success':
                result['status'] = 'success'
                result['execution_status'] = 'success'
                result['results_valid'] = execution_result['results_valid']
                result['result_count'] = execution_result['result_count']
                result['natural_language_answer'] = execution_result.get('natural_language_answer', '')
                result['validation_checks'] = execution_result.get('validation_checks', {})
                
                # Convert result data to DataFrame
                if execution_result.get('result_data'):
                    import pandas as pd
                    result['visualization_data'] = pd.DataFrame(execution_result['result_data'])
                else:
                    result['visualization_data'] = None
            else:
                result['status'] = 'error'
                result['error'] = execution_result.get('errors', ['Unknown execution error'])[0]
            
            result['execution_time'] = time.time() - start_time
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
        
        return result
    
    def _parse_response(self, response_text: str) -> dict:
        """Parse Gemini response into structured format"""
        # Remove markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()
        
        # Try to parse JSON
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                except:
                    # Fallback
                    result = {
                        "sql_query": "SELECT * FROM dataset LIMIT 100;",
                        "visualization": {"type": "table", "title": "Results"},
                        "explanation": "Fallback response"
                    }
            else:
                result = {
                    "sql_query": "SELECT * FROM dataset LIMIT 100;",
                    "visualization": {"type": "table", "title": "Results"},
                    "explanation": "Fallback response"
                }
        
        # Ensure required fields
        if "sql_query" not in result:
            result["sql_query"] = "SELECT * FROM dataset LIMIT 100;"
        if "visualization" not in result:
            result["visualization"] = {"type": "table", "title": "Results"}
        if "explanation" not in result:
            result["explanation"] = "Generated query"
        
        return result


if __name__ == "__main__":
    """Test query handler"""
    config = {
        'project_id': 'datacraft-data-pipeline',
        'dataset_id': 'datacraft_ml',
        'bucket_name': 'isha-retail-data',
        'region': 'us-central1',
        'model_name': 'gemini-2.5-pro',
        'table_name': 'orders_processed'
    }
    
    handler = QueryHandler(config)
    
    result = handler.process_query("What are the top 5 products by sales?", "orders")
    
    print(f"\nStatus: {result['status']}")
    print(f"Answer: {result.get('natural_language_answer')}")
    print(f"Rows: {result.get('result_count')}")