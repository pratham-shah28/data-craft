# scripts/query_executor.py
"""
Query Executor & Result Validator
Executes generated SQL queries and validates results make sense
Ensures the final answer is correct and meaningful
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime, date
import sys
import re

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging

from google.cloud import bigquery


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling date, datetime, and numpy types"""
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)


class QueryExecutor:
    """
    Execute SQL queries and validate results
    
    Features:
    1. Execute SQL on BigQuery
    2. Validate results match user intent
    3. Check if results are reasonable (not empty, not errors)
    4. Generate natural language answer from results
    5. Verify answer correctness
    """
    
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        service_account_path: Optional[str] = None
    ):
        """
        Initialize query executor
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            service_account_path: Optional path to service account
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        
        self.logger = setup_logging(self.__class__.__name__)
        
        # Setup GCP credentials
        from utils import setup_gcp_credentials
        try:
            setup_gcp_credentials(service_account_path, self.logger)
        except Exception as e:
            self.logger.warning(f"GCP credentials not configured: {e}")
        
        # Initialize BigQuery client
        try:
            self.bq_client = bigquery.Client(project=project_id)
            self.logger.info("✓ BigQuery client initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize BigQuery: {e}")
            raise
    
    def execute_and_validate(
        self,
        user_query: str,
        sql_query: str,
        table_name: str,
        visualization: Dict
    ) -> Dict:
        """
        Execute SQL query and validate results
        
        Args:
            user_query: Original user question
            sql_query: Generated SQL query
            table_name: Target table name
            visualization: Visualization configuration
            
        Returns:
            Dictionary with execution results and validation
        """
        self.logger.info(f"Executing query for: '{user_query}'")
        
        result = {
            "user_query": user_query,
            "sql_query": sql_query,
            "execution_status": "not_started",
            "results_valid": False,
            "result_count": 0,
            "result_data": None,
            "natural_language_answer": "",
            "validation_checks": {},
            "errors": []
        }
        
        try:
            # Step 1: Prepare SQL for BigQuery
            sql_bq = self._prepare_sql_for_bigquery(sql_query, table_name)
            self.logger.info(f"Prepared SQL: {sql_bq[:100]}...")
            
            # Step 2: Execute SQL
            df_result, execution_time = self._execute_sql(sql_bq)
            
            result["execution_status"] = "success"
            result["execution_time"] = execution_time
            result["result_count"] = len(df_result)
            result["result_data"] = df_result.to_dict('records') if len(df_result) <= 100 else df_result.head(100).to_dict('records')
            result["result_columns"] = list(df_result.columns)
            
            self.logger.info(f"✓ Query executed: {len(df_result)} rows returned in {execution_time:.2f}s")
            
            # Step 3: Validate results - updated 
            validation = self._validate_results(
                user_query=user_query,
                df_result=df_result,
                visualization=visualization 
            )

            result["validation_checks"] = validation
            result["results_valid"] = validation["overall_valid"]

            # Step 4: ALWAYS generate a natural-language answer
            nl_answer = self._generate_natural_language_answer(
                user_query=user_query,
                df_result=df_result,
                visualization=visualization
            )
            result["natural_language_answer"] = nl_answer

            if validation["overall_valid"]:
                self.logger.info(f"✓ Generated answer: {nl_answer[:100]}...")
            else:
                self.logger.warning("⚠ Results failed validation (soft) – returning answer anyway")

            
        except Exception as e:
            result["execution_status"] = "failed"
            result["errors"].append(str(e))
            self.logger.error(f"Query execution failed: {e}")
        
        return result
    
    def _prepare_sql_for_bigquery(
        self,
        sql: str,
        table_name: str
    ) -> str:
        """Convert generic SQL to BigQuery-specific SQL"""
        # Replace 'dataset' with actual table reference
        bq_table = f"`{self.project_id}.{self.dataset_id}.{table_name}`"
        
        # Handle different case variations and spacing
        # Replace "FROM dataset" (with various spacing/newlines)
        sql_bq = re.sub(
            r'FROM\s+dataset\b',
            f'FROM {bq_table}',
            sql,
            flags=re.IGNORECASE
        )
        
        # Also handle "from dataset" with different whitespace
        sql_bq = re.sub(
            r'from\s+dataset\b',
            f'from {bq_table}',
            sql_bq,
            flags=re.IGNORECASE
        )
        
        self.logger.info(f"Converted SQL for BigQuery table: {bq_table}")
        
        return sql_bq
    
    def _execute_sql(self, sql: str) -> Tuple[pd.DataFrame, float]:
        """Execute SQL on BigQuery and return results"""
        import time
        
        start_time = time.time()
        
        query_job = self.bq_client.query(sql)
        df = query_job.to_dataframe()
        
        execution_time = time.time() - start_time
        
        return df, execution_time
    
    def _validate_results(
        self,
        user_query: str,
        df_result: pd.DataFrame,
        visualization: Dict
    ) -> Dict:
        """
        Validate that results make sense
        
        Validation checks:
        1. Results not empty
        2. Result count reasonable
        3. Column types match visualization
        4. Values are within reasonable ranges
        5. Results answer the user's question
        """
        validation = {
            "overall_valid": True,
            "checks": {}
        }
        
        # Check 1: Results not empty
        if len(df_result) == 0:
            validation["checks"]["not_empty"] = {
                "passed": False,
                "message": "Query returned 0 rows - no data matches criteria"
            }
            validation["overall_valid"] = False
        else:
            validation["checks"]["not_empty"] = {
                "passed": True,
                "message": f"Query returned {len(df_result)} rows"
            }
        
        # Check 2: Reasonable result count
        result_count_check = self._check_result_count(len(df_result), user_query)
        validation["checks"]["reasonable_count"] = result_count_check
        if not result_count_check["passed"]:
            validation["overall_valid"] = False
        
        # Check 3: Column compatibility with visualization
        if len(df_result) > 0:
            viz_check = self._check_visualization_compatibility(df_result, visualization)
            validation["checks"]["visualization_compatible"] = viz_check
            if not viz_check["passed"]:
                validation["overall_valid"] = False
        
        # Check 4: Value reasonableness
        if len(df_result) > 0:
            value_check = self._check_value_reasonableness(df_result)
            validation["checks"]["values_reasonable"] = value_check
            if not value_check["passed"]:
                validation["overall_valid"] = False
        
        # Check 5: Intent alignment
        intent_check = self._check_intent_alignment(user_query, df_result)
        validation["checks"]["intent_aligned"] = intent_check
        if not intent_check["passed"]:
            validation["overall_valid"] = False
        
        return validation
    
    def _check_result_count(self, count: int, user_query: str) -> Dict:
        """Check if result count is reasonable"""
        user_lower = user_query.lower()
        
        # If user asked for "top N", check count matches
        import re
        top_match = re.search(r'top\s+(\d+)', user_lower)
        if top_match:
            expected_count = int(top_match.group(1))
            if count != expected_count and count < expected_count:
                return {
                    "passed": False,
                    "message": f"Expected {expected_count} results but got {count}"
                }
        
        # General reasonableness
        if count > 100000:
            return {
                "passed": False,
                "message": f"Too many results ({count}) - query might be too broad"
            }
        
        return {
            "passed": True,
            "message": f"Result count ({count}) is reasonable"
        }
    
    def _check_visualization_compatibility(
        self,
        df: pd.DataFrame,
        visualization: Dict
    ) -> Dict:
        """Check if results are compatible with chosen visualization"""
        viz_type = visualization.get('type', 'unknown')
        
        # Bar/Line/Scatter need at least 2 columns
        if viz_type in ['bar_chart', 'line_chart', 'scatter_plot']:
            if len(df.columns) < 2:
                return {
                    "passed": False,
                    "message": f"{viz_type} needs at least 2 columns, got {len(df.columns)}"
                }
        
        # Pie chart needs exactly 2 columns (labels, values)
        if viz_type == 'pie_chart':
            if len(df.columns) != 2:
                return {
                    "passed": False,
                    "message": f"Pie chart needs exactly 2 columns, got {len(df.columns)}"
                }
        
        # Check for numeric columns in visualizations
        if viz_type in ['bar_chart', 'line_chart', 'scatter_plot', 'pie_chart']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {
                    "passed": False,
                    "message": f"{viz_type} needs at least one numeric column"
                }
        
        return {
            "passed": True,
            "message": f"Results compatible with {viz_type}"
        }
    
    def _check_value_reasonableness(self, df: pd.DataFrame) -> Dict:
        """Check if values are reasonable (no infinities, extreme outliers)"""
        issues = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            # Check for infinities
            if np.isinf(df[col]).any():
                issues.append(f"Column '{col}' contains infinite values")
            
            # Check for extreme outliers (values > 1e10 or < -1e10)
            if (df[col].abs() > 1e10).any():
                issues.append(f"Column '{col}' has extreme values (>1e10)")
            
            # Check for all zeros
            if (df[col] == 0).all():
                issues.append(f"Column '{col}' is all zeros")
        
        if issues:
            return {
                "passed": False,
                "message": "; ".join(issues)
            }
        
        return {
            "passed": True,
            "message": "All values are within reasonable ranges"
        }
    
    def _check_intent_alignment(
        self,
        user_query: str,
        df_result: pd.DataFrame
    ) -> Dict:
        """Check if results align with user intent"""
        user_lower = user_query.lower()
        
        # If user asked for specific year
        if '2024' in user_query or '2023' in user_query or '2025' in user_query:
            # Check if results have date columns
            date_cols = df_result.select_dtypes(include=['datetime64']).columns
            if len(date_cols) == 0:
                # Check if any column name suggests dates
                has_date_column = any('date' in col.lower() or 'year' in col.lower() 
                                     for col in df_result.columns)
                if not has_date_column:
                    return {
                        "passed": False,
                        "message": "User asked about specific year but results have no date columns"
                    }
        
        # If user asked for "list all" or "show details"
        if any(phrase in user_lower for phrase in ['list all', 'show all', 'details', 'show me all']):
            # Should have multiple columns (not aggregated)
            if len(df_result.columns) < 3:
                return {
                    "passed": False,
                    "message": "User asked for details but results appear aggregated"
                }
        
        # If user asked for "total" or "sum"
        if any(phrase in user_lower for phrase in ['total', 'sum', 'how much', 'how many']):
            # Should have numeric results
            numeric_cols = df_result.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return {
                    "passed": False,
                    "message": "User asked for totals but results have no numeric columns"
                }
        
        return {
            "passed": True,
            "message": "Results align with user intent"
        }
    
    def _generate_natural_language_answer(
        self,
        user_query: str,
        df_result: pd.DataFrame,
        visualization: Dict
    ) -> str:
        """
        Generate natural language answer from query results
        
        Args:
            user_query: Original user question
            df_result: Query execution results
            visualization: Visualization type
            
        Returns:
            Natural language answer string
        """
        if len(df_result) == 0:
            return "No results found for your query."
        
        user_lower = user_query.lower()
        viz_type = visualization.get('type', 'table')
        
        # Handle different query types
        
        # Aggregation queries (total, sum, average)
        if any(word in user_lower for word in ['total', 'sum', 'how much', 'how many']):
            return self._answer_aggregation_query(user_query, df_result)
        
        # Ranking queries (top, best, highest)
        if any(word in user_lower for word in ['top', 'best', 'highest', 'worst', 'bottom']):
            return self._answer_ranking_query(user_query, df_result)
        
        # Comparison queries
        if any(word in user_lower for word in ['compare', 'versus', 'between']):
            return self._answer_comparison_query(user_query, df_result)
        
        # Trend queries
        if any(word in user_lower for word in ['trend', 'over time', 'growth']):
            return self._answer_trend_query(user_query, df_result)
        
        # Listing queries
        if any(word in user_lower for word in ['list', 'show all', 'display']):
            return self._answer_listing_query(user_query, df_result)
        
        # Default: summarize first few results
        return self._answer_general_query(user_query, df_result)
    
    def _answer_aggregation_query(self, user_query: str, df: pd.DataFrame) -> str:
        """Answer aggregation questions (total sales, average profit, etc.)"""
        # Find numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return f"The query returned {len(df)} results."
        
        # Get the main numeric value
        if len(df) == 1 and len(numeric_cols) > 0:
            # Single aggregated value
            col = numeric_cols[0]
            value = df[col].iloc[0]
            
            # Format based on column name
            if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'amount', 'price', 'cost']):
                return f"The total is ${value:,.2f}"
            elif 'count' in col.lower():
                return f"The count is {int(value):,}"
            elif 'discount' in col.lower():
                return f"The average discount is {value:.2%}"
            else:
                return f"The {col.replace('_', ' ')} is {value:,.2f}"
        
        # Multiple rows with aggregations (e.g., average per category)
        if len(df) > 1:
            cat_cols = df.select_dtypes(include=['object']).columns
            
            if len(cat_cols) > 0 and len(numeric_cols) > 0:
                category_col = cat_cols[0]
                value_col = numeric_cols[0]
                
                # Create summary of top items
                top_items = []
                for idx in range(min(3, len(df))):
                    cat = df[category_col].iloc[idx]
                    val = df[value_col].iloc[idx]
                    
                    if 'discount' in value_col.lower():
                        top_items.append(f"{cat} ({val:.1%})")
                    elif any(kw in value_col.lower() for kw in ['sales', 'revenue', 'amount']):
                        top_items.append(f"{cat} (${val:,.2f})")
                    else:
                        top_items.append(f"{cat} ({val:,.1f})")
                
                return f"Results by {category_col}: {', '.join(top_items)}. Total {len(df)} categories found."
        
        # Fallback
        col = numeric_cols[0]
        total = df[col].sum()
        avg = df[col].mean()
        
        return f"Found {len(df)} results with total of {total:,.2f} and average of {avg:,.2f}"
    
    def _answer_ranking_query(self, user_query: str, df: pd.DataFrame) -> str:
        """Answer ranking questions (top 10, best products, etc.)"""
        if len(df) == 0:
            return "No results found."
        
        # Find the key columns (typically first categorical and first numeric)
        cat_cols = df.select_dtypes(include=['object']).columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(cat_cols) > 0 and len(num_cols) > 0:
            label_col = cat_cols[0]
            value_col = num_cols[0]
            
            # Get top 3 for answer
            top_items = []
            for idx in range(min(3, len(df))):
                label = df[label_col].iloc[idx]
                value = df[value_col].iloc[idx]
                top_items.append(f"{label} (${value:,.2f})" if 'sales' in value_col.lower() else f"{label} ({value:,.0f})")
            
            return f"Top results: {', '.join(top_items)}. Total of {len(df)} results found."
        
        return f"Found {len(df)} ranked results."
    
    def _answer_comparison_query(self, user_query: str, df: pd.DataFrame) -> str:
        """Answer comparison questions"""
        if len(df) == 0:
            return "No results found for comparison."
        
        # Find comparison groups
        cat_cols = df.select_dtypes(include=['object']).columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(cat_cols) > 0 and len(num_cols) > 0:
            group_col = cat_cols[0]
            value_col = num_cols[0]
            
            # Get top and bottom
            max_row = df.loc[df[value_col].idxmax()]
            min_row = df.loc[df[value_col].idxmin()]
            
            max_label = max_row[group_col]
            max_value = max_row[value_col]
            min_label = min_row[group_col]
            min_value = min_row[value_col]
            
            return f"Comparison shows {max_label} has the highest ({max_value:,.2f}) while {min_label} has the lowest ({min_value:,.2f})"
        
        return f"Comparison shows {len(df)} different groups."
    
    def _answer_trend_query(self, user_query: str, df: pd.DataFrame) -> str:
        """Answer trend/time-series questions"""
        if len(df) < 2:
            return "Not enough data points to show a trend."
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(num_cols) > 0:
            value_col = num_cols[0]
            
            first_value = df[value_col].iloc[0]
            last_value = df[value_col].iloc[-1]
            
            change = last_value - first_value
            change_pct = (change / first_value * 100) if first_value != 0 else 0
            
            trend_direction = "increased" if change > 0 else "decreased"
            
            return f"The trend shows values {trend_direction} from {first_value:,.2f} to {last_value:,.2f} ({abs(change_pct):.1f}% change) over {len(df)} time periods."
        
        return f"Trend data shows {len(df)} time periods."
    
    def _answer_listing_query(self, user_query: str, df: pd.DataFrame) -> str:
        """Answer listing/detail questions"""
        if len(df) == 0:
            return "No records found matching your criteria."
        
        return f"Found {len(df)} records. Showing details in table view."
    
    def _answer_general_query(self, user_query: str, df: pd.DataFrame) -> str:
        """General answer for any query"""
        return f"Query returned {len(df)} rows with {len(df.columns)} columns."
    
    def execute_batch_queries(
        self,
        queries_and_sql: List[Dict],
        table_name: str
    ) -> List[Dict]:
        """
        Execute multiple queries and validate all results
        
        Args:
            queries_and_sql: List of dicts with user_query, sql_query, visualization
            table_name: Target table name
            
        Returns:
            List of execution results
        """
        self.logger.info(f"Executing batch of {len(queries_and_sql)} queries...")
        
        all_results = []
        
        for idx, item in enumerate(queries_and_sql, 1):
            self.logger.info(f"\n[{idx}/{len(queries_and_sql)}] {item['user_query']}")
            
            result = self.execute_and_validate(
                user_query=item['user_query'],
                sql_query=item['sql_query'],
                table_name=table_name,
                visualization=item['visualization']
            )
            
            all_results.append(result)
            
            # Log result
            if result['execution_status'] == 'success':
                if result['results_valid']:
                    self.logger.info(f"  ✓ Valid: {result['natural_language_answer'][:80]}...")
                else:
                    self.logger.warning(f"  ⚠ Executed but results invalid")
            else:
                self.logger.error(f"  ✗ Failed: {result['errors']}")
        
        # Summary
        successful = sum(1 for r in all_results if r['execution_status'] == 'success')
        valid = sum(1 for r in all_results if r['results_valid'])
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Batch Execution Summary:")
        self.logger.info(f"  Total Queries: {len(queries_and_sql)}")
        self.logger.info(f"  Executed Successfully: {successful}")
        self.logger.info(f"  Results Valid: {valid}")
        self.logger.info(f"{'='*60}")
        
        return all_results
    
    def save_execution_results(
        self,
        results: List[Dict],
        output_dir: str
    ) -> str:
        """
        Save execution results to JSON file
        
        Args:
            results: List of execution results
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = output_path / f"execution_results_{timestamp}.json"
        
        summary = {
            "execution_timestamp": datetime.now().isoformat(),
            "total_queries": len(results),
            "successful_executions": sum(1 for r in results if r['execution_status'] == 'success'),
            "valid_results": sum(1 for r in results if r['results_valid']),
            "results": results
        }
        
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2, cls=JSONEncoder)
        
        self.logger.info(f"✓ Saved execution results: {result_file}")
        
        return str(result_file)


# ========================================
# STANDALONE TESTING
# ========================================

if __name__ == "__main__":
    """
    Test script
    Usage: python query_executor.py
    """
    
    print("\n" + "=" * 60)
    print("TESTING QUERY EXECUTOR")
    print("=" * 60 + "\n")
    
    # Mock test (without actual BigQuery)
    print("Note: Full testing requires BigQuery credentials")
    print("This test demonstrates the validation logic\n")
    
    # Create mock results
    mock_results = pd.DataFrame({
        'product_name': ['Product A', 'Product B', 'Product C'],
        'total_sales': [15000.50, 12000.00, 9500.75]
    })
    
    try:
        executor = QueryExecutor(
            project_id="datacraft-data-pipeline",
            dataset_id="datacraft_ml"
        )
        
        # Test validation logic
        print("1. Testing result validation...")
        validation = executor._validate_results(
            user_query="What are the top 3 products by sales?",
            df_result=mock_results,
            visualization={"type": "bar_chart", "title": "Top Products"}
        )
        
        print(f"\n   Validation Results:")
        print(f"   Overall Valid: {validation['overall_valid']}")
        for check_name, check_result in validation['checks'].items():
            status = "✓" if check_result['passed'] else "✗"
            print(f"   {status} {check_name}: {check_result['message']}")
        
        # Test natural language answer generation
        print("\n2. Testing natural language answer generation...")
        answer = executor._generate_natural_language_answer(
            user_query="What are the top 3 products by sales?",
            df_result=mock_results,
            visualization={"type": "bar_chart"}
        )
        
        print(f"\n   Generated Answer:")
        print(f"   '{answer}'")
        
        print("\n" + "=" * 60)
        print("✓ QUERY EXECUTOR TEST PASSED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)