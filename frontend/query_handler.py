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
import logging


class QueryHandler:
    """
    Handle user queries end-to-end
    
    Flow:
    1. Get dataset metadata
    2. Build prompt with context
    3. Call Gemini model
    4. Parse response (handles info messages)
    5. Apply runtime type conversions (STRING → NUMERIC)
    6. Execute SQL on BigQuery
    7. Validate results
    8. Return complete answer with visualization
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
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.metadata_manager = MetadataManager(self.project_id, self.dataset_id)
        self.query_executor = QueryExecutor(self.project_id, self.dataset_id)
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.region)
        
        # Load best model configuration (from pipeline)
        self.generation_config = self._load_best_model_config()
        
        # ✅ Cache for numeric string columns (avoid repeated detection)
        self._numeric_columns_cache = {}
    
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
    
    def _get_numeric_string_columns(self, dataset_name: str) -> dict:
        """Detect STRING columns that contain numeric data"""
        # Check cache first
        if dataset_name in self._numeric_columns_cache:
            return self._numeric_columns_cache[dataset_name]
        
        self.logger.info(f"Detecting numeric string columns for {dataset_name}...")
        
        full_table_name = f"`{self.project_id}.{self.dataset_id}.{dataset_name}_processed`"
        
        try:
            from google.cloud import bigquery
            client = bigquery.Client(project=self.project_id)
            
            # ✅ FIX: Remove the problematic LIKE clause
            schema_query = f"""
            SELECT column_name
            FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{dataset_name}_processed'
            AND data_type = 'STRING'
            """
            
            results = client.query(schema_query).result()
            string_columns = [row.column_name for row in results]
            
            # Filter out metadata columns in Python instead
            string_columns = [col for col in string_columns if not col.startswith('_')]
            
            numeric_columns = {}
            
            # Test each STRING column
            for col in string_columns[:20]:
                # ✅ FIX: Escape column name properly and simplify regex
                test_query = f"""
                SELECT 
                    COUNTIF(`{col}` IS NOT NULL) as non_null,
                    COUNTIF(SAFE_CAST(REGEXP_REPLACE(`{col}`, r'[$,€£¥\\s%]', '') AS FLOAT64) IS NOT NULL) as convertible
                FROM {full_table_name}
                LIMIT 100
                """
                
                try:
                    test_result = client.query(test_query).result()
                    row = list(test_result)[0]
                    
                    if row.non_null > 0:
                        success_rate = row.convertible / row.non_null
                        
                        if success_rate > 0.8:
                            numeric_columns[col] = 'FLOAT64'
                            self.logger.info(f"✓ {col}: {success_rate:.0%} numeric")
                except Exception as e:
                    self.logger.debug(f"Could not test column {col}: {str(e)}")
                    continue
            
            # Cache the result
            self._numeric_columns_cache[dataset_name] = numeric_columns
            
            return numeric_columns
            
        except Exception as e:
            self.logger.warning(f"Column detection failed: {str(e)}")
            return {}
    
    def _wrap_sql_with_type_conversions(self, sql_query: str, dataset_name: str) -> str:
        """Automatically convert STRING to NUMERIC in SQL at runtime"""
        numeric_columns = self._get_numeric_string_columns(dataset_name)
        
        if not numeric_columns:
            return sql_query
        
        self.logger.info(f"Applying runtime type conversions for: {list(numeric_columns.keys())}")
        
        # Helper function to create conversion expression
        def convert_col(col_name):
            # ✅ FIX: Only apply REGEXP_REPLACE if it's a STRING column
            # Check if column appears to already be numeric in the query
            return f"SAFE_CAST(REGEXP_REPLACE(CAST(`{col_name}` AS STRING), r'[$,€£¥\\s%]', '') AS FLOAT64)"
        
        # Apply conversions for each numeric string column
        for col_name in numeric_columns.keys():
            converted_expr = convert_col(col_name)
            
            # Pattern 1: Aggregate functions
            agg_functions = ['SUM', 'AVG', 'MAX', 'MIN', 'COUNT']
            for func in agg_functions:
                # ✅ Match both backticked and non-backticked versions
                pattern = rf'\b{func}\s*\(\s*(?:CAST\s*\(\s*)?`?{re.escape(col_name)}`?(?:\s+AS\s+\w+\s*\))?\s*\)'
                replacement = f'{func}({converted_expr})'
                sql_query = re.sub(pattern, replacement, sql_query, flags=re.IGNORECASE)
            
            # Pattern 2: WHERE/HAVING comparisons
            comparison_ops = ['>=', '<=', '!=', '<>', '>', '<', '=']
            for op in comparison_ops:
                escaped_op = re.escape(op)
                pattern = rf'`?{col_name}`?\s*{escaped_op}\s*(\d+\.?\d*)'
                replacement = rf'{converted_expr} {op} \1'
                sql_query = re.sub(pattern, replacement, sql_query)
            
            # Pattern 3: Arithmetic operations (+, -, *, /)
            arithmetic_ops = [r'\+', r'\-', r'\*', r'/']
            for op in arithmetic_ops:
                pattern = rf'`?{col_name}`?\s*{op}\s*(\d+\.?\d*)'
                clean_op = op.replace('\\', '')
                replacement = rf'{converted_expr} {clean_op} \1'
                sql_query = re.sub(pattern, replacement, sql_query)
            
            # Pattern 4: ORDER BY
            pattern = rf'\bORDER\s+BY\s+`?{col_name}`?(?:\s+(ASC|DESC))?'
            def replace_order(match):
                direction = match.group(1) if match.group(1) else ''
                return f'ORDER BY {converted_expr} {direction}'.strip()
            sql_query = re.sub(pattern, replace_order, sql_query, flags=re.IGNORECASE)
            
            # Pattern 5: BETWEEN
            pattern = rf'`?{col_name}`?\s+BETWEEN\s+(\d+\.?\d*)\s+AND\s+(\d+\.?\d*)'
            replacement = rf'{converted_expr} BETWEEN \1 AND \2'
            sql_query = re.sub(pattern, replacement, sql_query, flags=re.IGNORECASE)
        
        return sql_query
    
    def process_query(self, user_query: str, dataset_name: str) -> dict:
        """
        Process user query end-to-end
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
            
            # ✅ ADD EXPLICIT TABLE NAME TO CONTEXT
            full_table_name = f"`{self.project_id}.{self.dataset_id}.{dataset_name}_processed`"
            
            if full_table_name not in llm_context:
                llm_context = f"TABLE NAME: {full_table_name}\n\n{llm_context}"
            
            # Step 2: Build prompt
            prompt = build_prompt(user_query, llm_context, FEW_SHOT_EXAMPLES)
            
            prompt = prompt.replace(
                "Generate a SQL query for BigQuery",
                f"Generate a SQL query for BigQuery. ALWAYS use the full table name: {full_table_name} in your FROM clause. NEVER use placeholder names like 'dataset' or 'table_name'."
            )
            
            self.logger.info(f"Processing query: '{user_query}' for dataset: {dataset_name}")
            self.logger.info(f"Using table: {full_table_name}")
            
            # Step 3: Call Gemini model
            model = GenerativeModel(self.model_name, generation_config=self.generation_config)
            response = model.generate_content(prompt)
            response_text = response.text
            
            self.logger.info(f"LLM Response received (length: {len(response_text)} chars)")
            
            # Step 4: Parse response
            parsed = self._parse_response(response_text, dataset_name)
            
            # ✅ CHECK IF THIS IS AN INFORMATIONAL MESSAGE - RETURN IMMEDIATELY
            if parsed.get('is_message'):
                self.logger.info("LLM returned a message instead of SQL")
                return {
                    'status': 'info',
                    'message': parsed['message'],
                    'user_query': user_query,
                    'dataset_name': dataset_name,
                    'execution_time': time.time() - start_time
                }
            
            # ✅ ADD SAFETY CHECK: Ensure sql_query exists and is not None
            if not parsed.get('sql_query'):
                self.logger.error("No SQL query in parsed response")
                return {
                    'status': 'error',
                    'error': 'Failed to generate SQL query. Please try rephrasing your question.',
                    'user_query': user_query,
                    'dataset_name': dataset_name,
                    'execution_time': time.time() - start_time
                }
            
            # ✅ NOW SAFE TO USE .strip()
            sql_query = parsed['sql_query'].strip()
            
            # ✅ FORCE REPLACE PLACEHOLDER TABLE NAMES
            placeholder_replacements = [
                (r'\bFROM\s+dataset\b', f'FROM {full_table_name}'),
                (r'\bJOIN\s+dataset\b', f'JOIN {full_table_name}'),
                (r'\bFROM\s+`?dataset`?\b', f'FROM {full_table_name}'),
                (r'\bFROM\s+table\b', f'FROM {full_table_name}'),
                (r'\bFROM\s+your_table\b', f'FROM {full_table_name}'),
            ]
            
            for pattern, replacement in placeholder_replacements:
                sql_query = re.sub(pattern, replacement, sql_query, flags=re.IGNORECASE)
            
            # ✅ FIX DATE COMPARISONS
            sql_query = self._fix_date_comparisons(sql_query)
            
            # ✅ APPLY RUNTIME TYPE CONVERSIONS (STRING → NUMERIC)
            sql_query = self._wrap_sql_with_type_conversions(sql_query, dataset_name)
            
            parsed['sql_query'] = sql_query
            
            # Validate parsed response before execution
            if not sql_query or 'dataset' in sql_query.lower():
                if re.search(r'\bFROM\s+dataset\b', sql_query, re.IGNORECASE):
                    self.logger.error(f"Still has placeholder after replacement: {sql_query}")
                    raise ValueError(
                        f"Invalid SQL generated. Unable to replace placeholder table name. "
                        f"Generated SQL: {sql_query}"
                    )
            
            result['sql_query'] = sql_query
            result['visualization'] = parsed['visualization']
            result['explanation'] = parsed['explanation']
            
            self.logger.info(f"Parsed SQL successfully. Executing query...")
            
            # Step 5: Execute SQL on BigQuery
            execution_result = self.query_executor.execute_and_validate(
                user_query=user_query,
                sql_query=sql_query,
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
                
                self.logger.info(f"Query executed successfully. Rows returned: {result['result_count']}")
            else:
                result['status'] = 'error'
                result['error'] = execution_result.get('errors', ['Unknown execution error'])[0]
                self.logger.error(f"Query execution failed: {result['error']}")
            
            result['execution_time'] = time.time() - start_time
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
            self.logger.error(f"Query processing failed: {str(e)}")
        
        return result
    
    def _parse_response(self, response_text: str, dataset_name: str) -> dict:
        """
        Parse Gemini response into structured format
        Handles both valid SQL responses and explanatory messages
        """
        original_response = response_text
        
        # Remove markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()
        
        # ✅ Quick check: If starts with comment, it's a message
        if response_text.startswith('--'):
            message = response_text.replace('--', '').strip()
            return {
                'is_message': True,
                'message': message,
                'sql_query': None,
                'visualization': {'type': 'none'},
                'explanation': message
            }
        
        # Try to parse JSON
        try:
            result = json.loads(response_text)
            
            # ✅ KEY FIX: Check if sql_query is explicitly null (as instructed in prompt)
            if result.get('sql_query') is None:
                self.logger.info("LLM returned null sql_query (personal/non-data question)")
                return {
                    'is_message': True,
                    'message': result.get('explanation', 'Unable to process this query.'),
                    'sql_query': None,
                    'visualization': result.get('visualization', {'type': 'none'}),
                    'explanation': result.get('explanation', 'Unable to process this query.')
                }
            
            # ✅ Check if sql_query is empty string
            sql_query = result.get('sql_query', '').strip()
            if not sql_query:
                self.logger.info("LLM returned empty sql_query")
                return {
                    'is_message': True,
                    'message': result.get('explanation', 'Unable to process this query.'),
                    'sql_query': None,
                    'visualization': result.get('visualization', {'type': 'none'}),
                    'explanation': result.get('explanation', 'Unable to process this query.')
                }
            
            # ✅ Check if sql_query contains conversational text instead of SQL
            conversational_patterns = [
                'cannot answer personal questions',
                'do not have personal preferences',
                'as a data analyst assistant',
                'my purpose is to',
                'i can only answer'
            ]
            
            sql_lower = sql_query.lower()
            if any(pattern in sql_lower for pattern in conversational_patterns):
                self.logger.info("SQL field contains conversational text")
                return {
                    'is_message': True,
                    'message': result.get('explanation', sql_query),
                    'sql_query': None,
                    'visualization': result.get('visualization', {'type': 'none'}),
                    'explanation': result.get('explanation', sql_query)
                }
            
            # ✅ Check if it's valid SQL (must contain SELECT)
            if 'SELECT' not in sql_query.upper():
                self.logger.info("SQL doesn't contain SELECT keyword")
                return {
                    'is_message': True,
                    'message': result.get('explanation', 'Unable to generate valid SQL.'),
                    'sql_query': None,
                    'visualization': result.get('visualization', {'type': 'none'}),
                    'explanation': result.get('explanation', 'Unable to generate valid SQL.')
                }
            
            # Valid SQL query - mark as not a message
            result['is_message'] = False
            self.logger.info("Successfully parsed valid SQL query")
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parse failed: {str(e)}. Attempting extraction...")
            
            # Try to extract JSON from text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    
                    # Apply same checks as above
                    if result.get('sql_query') is None or not result.get('sql_query', '').strip():
                        return {
                            'is_message': True,
                            'message': result.get('explanation', 'Unable to process this query.'),
                            'sql_query': None,
                            'visualization': result.get('visualization', {'type': 'none'}),
                            'explanation': result.get('explanation', 'Unable to process this query.')
                        }
                    
                    result['is_message'] = False
                    
                except Exception as e2:
                    self.logger.error(f"JSON extraction failed: {str(e2)}")
                    raise ValueError(f"Failed to parse LLM response: {str(e2)}")
            else:
                self.logger.error("No JSON found in response")
                raise ValueError("LLM response does not contain valid JSON")
        
        # Validate required fields (only if it's not a message)
        if not result.get('is_message', False):
            if 'visualization' not in result:
                result['visualization'] = {
                    "type": "table",
                    "title": "Query Results",
                    "x_axis": None,
                    "y_axis": None
                }
            
            if 'explanation' not in result:
                result['explanation'] = "Query generated based on user request"
        
        return result
    
    
    def _fix_date_comparisons(self, sql_query: str) -> str:
        """
        Fix date comparison and parsing issues in SQL
        
        Args:
            sql_query: Original SQL query
            
        Returns:
            Fixed SQL query
        """
        original_query = sql_query
        
        # Pattern 1: Fix PARSE_DATE with any format - replace with CAST
        if 'PARSE_DATE' in sql_query:
            sql_query = re.sub(
                r"PARSE_DATE\('[^']+',\s*(\w+)\)",
                r"CAST(\1 AS DATE)",
                sql_query
            )
            self.logger.info("Replaced PARSE_DATE with CAST")
        
        # Pattern 2: Fix PARSE_TIMESTAMP with any format - replace with CAST
        if 'PARSE_TIMESTAMP' in sql_query:
            sql_query = re.sub(
                r"PARSE_TIMESTAMP\('[^']+',\s*(\w+)\)",
                r"CAST(\1 AS DATE)",
                sql_query
            )
            self.logger.info("Replaced PARSE_TIMESTAMP with CAST")
        
        # Pattern 3: Fix DATE_TRUNC with PARSE_DATE/PARSE_TIMESTAMP
        # Replace DATE_TRUNC(PARSE_DATE(...), MONTH) with DATE_TRUNC(CAST(...), MONTH)
        sql_query = re.sub(
            r"DATE_TRUNC\(PARSE_DATE\('[^']+',\s*(\w+)\),\s*(\w+)\)",
            r"DATE_TRUNC(CAST(\1 AS DATE), \2)",
            sql_query
        )
        
        sql_query = re.sub(
            r"DATE_TRUNC\(PARSE_TIMESTAMP\('[^']+',\s*(\w+)\),\s*(\w+)\)",
            r"DATE_TRUNC(CAST(\1 AS DATE), \2)",
            sql_query
        )
        
        # Pattern 4: Fix date comparisons like: column >= '2024-01-01'
        # Make sure the column is cast to DATE for comparison
        date_pattern = r"(\w+)\s*(>=|<=|>|<|=)\s*'(\d{4}-\d{2}-\d{2})'"
        
        def replace_date_comparison(match):
            column = match.group(1)
            operator = match.group(2)
            date_value = match.group(3)
            
            # Use CAST instead of PARSE_DATE for more flexibility
            return f"CAST({column} AS DATE) {operator} DATE '{date_value}'"
        
        sql_query = re.sub(date_pattern, replace_date_comparison, sql_query)
        
        if sql_query != original_query:
            self.logger.info("Applied date handling fixes to SQL query")
        
        return sql_query


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