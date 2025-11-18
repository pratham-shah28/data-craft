# scripts/model_pipeline.py
"""
Model Pipeline Helper Functions for Airflow
All functions designed to work with Airflow operators via XCom
"""

import json
import pandas as pd
import duckdb
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys
import re
import logging

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging
from prompts import build_prompt, validate_response_format, FEW_SHOT_EXAMPLES


class QueryVisualizationHelper:
    """
    Helper functions for query generation pipeline
    Designed to work with Airflow tasks (no direct Vertex AI calls)
    """
    
    def __init__(self):
        self.logger = setup_logging(self.__class__.__name__)
    
    def build_gemini_prompt(
        self, 
        user_query: str, 
        dataset_context: str,
        few_shot_examples: str = FEW_SHOT_EXAMPLES
    ) -> str:
        """
        Build prompt for Gemini (to be used by Airflow Vertex AI operator)
        
        Args:
            user_query: User's natural language question
            dataset_context: Rich context about dataset from metadata
            few_shot_examples: Few-shot examples for prompting
            
        Returns:
            Complete prompt string
            
        Example:
            >>> helper = QueryVisualizationHelper()
            >>> prompt = helper.build_gemini_prompt(
            ...     "What are top products?",
            ...     metadata['llm_context']
            ... )
        """
        self.logger.info(f"Building prompt for query: '{user_query}'")
        
        prompt = build_prompt(user_query, dataset_context, few_shot_examples)
        
        self.logger.info(f"✓ Built prompt ({len(prompt)} characters)")
        
        return prompt
    
    def parse_gemini_response(self, response_text: str) -> Dict:
        """
        Parse JSON response from Gemini
        
        Handles various formats:
        - Clean JSON
        - JSON wrapped in markdown code blocks
        - JSON with extra text
        
        Args:
            response_text: Raw text response from Gemini
            
        Returns:
            Parsed dictionary with sql_query, visualization, explanation
        """
        self.logger.info("Parsing Gemini response")
        
        # Remove markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        response_text = response_text.strip()
        
        # Try direct JSON parse
        try:
            result = json.loads(response_text)
            self.logger.info("✓ Parsed JSON successfully")
            return result
        except json.JSONDecodeError:
            self.logger.warning("Direct JSON parse failed, trying extraction")
        
        # Try to extract JSON from text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                self.logger.info("✓ Extracted and parsed JSON")
                return result
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to extract SQL at minimum
        return self._extract_sql_fallback(response_text)
    
    def _extract_sql_fallback(self, text: str) -> Dict:
        """Extract SQL query as fallback when JSON parsing fails"""
        self.logger.warning("Using SQL fallback extraction")
        
        sql_patterns = [
            r'SELECT\s+.*?FROM\s+dataset.*?(?:;|$)',
            r'SELECT\s+.*?FROM\s+.*?(?:;|$)'
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sql = match.group(0).strip()
                return {
                    "sql_query": sql,
                    "visualization": {
                        "type": "table",
                        "title": "Query Results"
                    },
                    "explanation": "Generated query (visualization defaulted to table)"
                }
        
        # Ultimate fallback
        return {
            "sql_query": "SELECT * FROM dataset LIMIT 100;",
            "visualization": {
                "type": "table",
                "title": "Data Preview"
            },
            "explanation": "Fallback query to show data preview"
        }
    
    def validate_and_fix_response(self, response: Dict) -> Dict:
        """
        Validate response format and apply fixes if needed
        
        Args:
            response: Parsed response dictionary
            
        Returns:
            Validated and potentially fixed response
        """
        self.logger.info("Validating response format")
        
        # Check if response is valid
        if validate_response_format(response):
            self.logger.info("✓ Response format is valid")
            return response
        
        # Try to fix common issues
        fixed_response = response.copy()
        
        # Ensure required fields exist
        if "sql_query" not in fixed_response:
            fixed_response["sql_query"] = "SELECT * FROM dataset LIMIT 100;"
        
        if "visualization" not in fixed_response:
            fixed_response["visualization"] = {
                "type": "table",
                "title": "Query Results"
            }
        elif "type" not in fixed_response["visualization"]:
            fixed_response["visualization"]["type"] = "table"
        elif "title" not in fixed_response["visualization"]:
            fixed_response["visualization"]["title"] = "Query Results"
        
        if "explanation" not in fixed_response:
            fixed_response["explanation"] = "Query generated successfully"
        
        self.logger.info("✓ Response format fixed")
        return fixed_response
    
    def execute_sql_query(
        self, 
        sql: str, 
        df: pd.DataFrame,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Execute SQL query using DuckDB (in-memory)
        
        Args:
            sql: SQL query to execute (should use 'dataset' as table name)
            df: DataFrame to query against
            limit: Maximum rows to return (safety limit)
            
        Returns:
            DataFrame with query results
        """
        self.logger.info("Executing SQL query with DuckDB")
        
        # Create in-memory DuckDB connection
        conn = duckdb.connect(database=':memory:')
        
        try:
            # Register DataFrame as 'dataset' table
            conn.register('dataset', df)
            
            # Add LIMIT if not present (safety)
            if 'LIMIT' not in sql.upper():
                sql = sql.rstrip(';') + f' LIMIT {limit};'
            
            # Execute query
            result = conn.execute(sql).df()
            
            self.logger.info(f"✓ Query executed successfully ({len(result)} rows returned)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"SQL execution failed: {str(e)}")
            raise ValueError(f"SQL execution failed: {str(e)}")
            
        finally:
            conn.close()
    
    def validate_sql_syntax(self, sql: str) -> Tuple[bool, str]:
        """
        Validate SQL syntax without executing
        
        Args:
            sql: SQL query string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            conn = duckdb.connect(database=':memory:')
            # Create dummy table to test syntax
            conn.execute("CREATE TABLE dataset (id INTEGER)")
            conn.execute(f"EXPLAIN {sql}")
            conn.close()
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def prepare_sql_for_bigquery(
        self, 
        sql: str, 
        project_id: str,
        dataset_id: str,
        table_name: str
    ) -> str:
        """
        Convert generic SQL to BigQuery-specific SQL
        
        Args:
            sql: Generic SQL with 'dataset' as table name
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            table_name: Actual table name
            
        Returns:
            BigQuery-compatible SQL
        """
        # Replace generic 'dataset' with full BigQuery table reference
        bq_table = f"`{project_id}.{dataset_id}.{table_name}`"
        sql_bq = sql.replace('FROM dataset', f'FROM {bq_table}')
        
        self.logger.info(f"✓ Converted SQL for BigQuery")
        
        return sql_bq
    
    def format_response_for_storage(
        self,
        user_query: str,
        sql_query: str,
        visualization: Dict,
        explanation: str,
        execution_success: bool,
        result_rows: int = 0,
        error_message: str = ""
    ) -> Dict:
        """
        Format response for storage in BigQuery query log
        
        Returns:
            Dictionary ready for BigQuery insertion
        """
        from datetime import datetime
        
        return {
            "query_id": f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "user_query": user_query,
            "generated_sql": sql_query,
            "visualization": json.dumps(visualization),
            "explanation": explanation,
            "execution_success": execution_success,
            "result_rows": result_rows,
            "error_message": error_message,
            "created_at": datetime.now().isoformat()
        }


# ========================================
# AIRFLOW-COMPATIBLE FUNCTIONS
# ========================================

def build_prompt_task(**context):
    """
    Airflow task: Build prompt for Gemini
    
    Usage in DAG:
        build_prompt = PythonOperator(
            task_id='build_prompt',
            python_callable=build_prompt_task
        )
    """
    helper = QueryVisualizationHelper()
    
    # Get inputs from XCom or DAG config
    ti = context['ti']
    user_query = context['dag_run'].conf.get('user_query', 'Show me sales data')
    
    # Get metadata from previous task
    metadata_result = ti.xcom_pull(task_ids='load_metadata')
    
    if not metadata_result:
        raise ValueError("No metadata found from previous task")
    
    # Extract LLM context
    llm_context = metadata_result.get('llm_context', '')
    
    # Build prompt
    prompt = helper.build_gemini_prompt(user_query, llm_context)
    
    # Store in XCom for next task
    ti.xcom_push(key='prompt', value=prompt)
    ti.xcom_push(key='user_query', value=user_query)
    
    return {"status": "success", "prompt_length": len(prompt)}


def parse_gemini_response_task(**context):
    """
    Airflow task: Parse Gemini response
    
    Usage in DAG:
        parse_response = PythonOperator(
            task_id='parse_response',
            python_callable=parse_gemini_response_task
        )
    """
    helper = QueryVisualizationHelper()
    
    ti = context['ti']
    
    # Get Gemini response from previous task
    gemini_response = ti.xcom_pull(task_ids='generate_query')
    
    if not gemini_response:
        raise ValueError("No response from Gemini")
    
    # Extract text from response (format depends on Airflow operator)
    if isinstance(gemini_response, list) and len(gemini_response) > 0:
        response_text = gemini_response[0]
    elif isinstance(gemini_response, dict) and 'text' in gemini_response:
        response_text = gemini_response['text']
    else:
        response_text = str(gemini_response)
    
    # Parse response
    parsed = helper.parse_gemini_response(response_text)
    
    # Validate and fix if needed
    validated = helper.validate_and_fix_response(parsed)
    
    # Store in XCom
    ti.xcom_push(key='sql_query', value=validated['sql_query'])
    ti.xcom_push(key='visualization', value=validated['visualization'])
    ti.xcom_push(key='explanation', value=validated['explanation'])
    
    return validated


def execute_sql_task(**context):
    """
    Airflow task: Execute generated SQL query
    
    Usage in DAG:
        execute_sql = PythonOperator(
            task_id='execute_sql',
            python_callable=execute_sql_task
        )
    """
    from google.cloud import bigquery
    
    helper = QueryVisualizationHelper()
    ti = context['ti']
    
    # Get SQL from previous task
    sql_query = ti.xcom_pull(task_ids='parse_response', key='sql_query')
    
    # Get configuration
    project_id = context['dag'].params.get('project_id', 'datacraft-data-pipeline')
    dataset_id = context['dag'].params.get('dataset_id', 'datacraft_ml')
    table_name = context['dag'].params.get('table_name', 'orders_processed')
    
    # Convert SQL for BigQuery
    sql_bq = helper.prepare_sql_for_bigquery(
        sql_query, 
        project_id, 
        dataset_id, 
        table_name
    )
    
    # Execute on BigQuery
    client = bigquery.Client(project=project_id)
    
    try:
        query_job = client.query(sql_bq)
        result_df = query_job.to_dataframe()
        
        # Store results
        ti.xcom_push(key='result_rows', value=len(result_df))
        ti.xcom_push(key='execution_success', value=True)
        ti.xcom_push(key='sample_results', value=result_df.head(10).to_dict('records'))
        
        return {
            "status": "success",
            "rows": len(result_df),
            "columns": len(result_df.columns)
        }
        
    except Exception as e:
        ti.xcom_push(key='execution_success', value=False)
        ti.xcom_push(key='error_message', value=str(e))
        raise


# ========================================
# STANDALONE TESTING
# ========================================

if __name__ == "__main__":
    """
    Local testing script
    Usage: python model_pipeline.py
    """
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import ModelDataLoader
    from metadata_manager import MetadataManager
    
    # Configuration
    PROJECT_ID = "datacraft-data-pipeline"
    BUCKET_NAME = "isha-retail-data"
    DATASET_ID = "datacraft_ml"
    DATASET_NAME = "orders"
    
    try:
        print("\n" + "=" * 60)
        print("TESTING MODEL PIPELINE HELPER")
        print("=" * 60 + "\n")
        
        # Initialize helper
        print("1. Initializing helper...")
        helper = QueryVisualizationHelper()
        
        # Load metadata
        print("\n2. Loading metadata...")
        manager = MetadataManager(PROJECT_ID, DATASET_ID)
        metadata_result = manager.get_metadata(DATASET_NAME)
        
        if not metadata_result:
            print("   No metadata found! Run metadata_manager.py first")
            sys.exit(1)
        
        dataset_context = metadata_result['llm_context']
        
        # Test prompt building
        print("\n3. Testing prompt building...")
        test_query = "What are the top 5 products by sales?"
        prompt = helper.build_gemini_prompt(test_query, dataset_context)
        print(f"   Prompt length: {len(prompt)} characters")
        print(f"   Preview: {prompt[:200]}...")
        
        # Test response parsing
        print("\n4. Testing response parsing...")
        mock_response = '''```json
        {
            "sql_query": "SELECT product_name, SUM(sales) as total_sales FROM dataset GROUP BY product_name ORDER BY total_sales DESC LIMIT 5",
            "visualization": {
                "type": "bar_chart",
                "x_axis": "product_name",
                "y_axis": "total_sales",
                "title": "Top 5 Products by Sales"
            },
            "explanation": "Shows top 5 products by total sales"
        }
```'''
        
        parsed = helper.parse_gemini_response(mock_response)
        print(f"   Parsed SQL: {parsed['sql_query'][:80]}...")
        print(f"   Viz Type: {parsed['visualization']['type']}")
        
        # Test validation
        print("\n5. Testing validation...")
        validated = helper.validate_and_fix_response(parsed)
        print(f"   Validation: {'✓ Passed' if validate_response_format(validated) else '✗ Failed'}")
        
        # Test SQL conversion
        print("\n6. Testing SQL conversion for BigQuery...")
        sql_bq = helper.prepare_sql_for_bigquery(
            parsed['sql_query'],
            PROJECT_ID,
            DATASET_ID,
            f"{DATASET_NAME}_processed"
        )
        print(f"   BigQuery SQL: {sql_bq[:100]}...")
        
        # Test SQL execution with DuckDB
        print("\n7. Testing SQL execution with DuckDB...")
        loader = ModelDataLoader(BUCKET_NAME, PROJECT_ID, DATASET_ID)
        
        try:
            df = loader.query_bigquery_table(f"{DATASET_NAME}_processed", limit=1000)
        except:
            df = loader.load_processed_data_from_gcs(DATASET_NAME, 'validated')
        
        result_df = helper.execute_sql_query(parsed['sql_query'], df)
        print(f"   Results: {len(result_df)} rows")
        if len(result_df) > 0:
            print(f"   Preview:\n{result_df.head(3)}")
        
        # Test response formatting
        print("\n8. Testing response formatting for storage...")
        formatted = helper.format_response_for_storage(
            user_query=test_query,
            sql_query=parsed['sql_query'],
            visualization=parsed['visualization'],
            explanation=parsed['explanation'],
            execution_success=True,
            result_rows=len(result_df)
        )
        print(f"   Query ID: {formatted['query_id']}")
        print(f"   Status: {'Success' if formatted['execution_success'] else 'Failed'}")
        
        print("\n" + "=" * 60)
        print("✓ MODEL PIPELINE HELPER TEST PASSED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


