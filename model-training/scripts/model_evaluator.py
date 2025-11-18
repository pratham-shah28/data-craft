# scripts/model_evaluator.py
"""
Model Evaluation for Query & Visualization Models
Evaluates LLM performance on SQL generation and visualization tasks
Generates comprehensive evaluation reports for each model

ENHANCED with User Intent Matching
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import re
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging

import duckdb
from google.cloud import bigquery


class ModelEvaluator:
    """
    Comprehensive model evaluation for SQL generation and visualization tasks
    
    Metrics Evaluated:
    ✅ BASIC METRICS:
    - SQL Syntax Validity (can the query be parsed?)
    - SQL Executability (can the query run without errors?)
    - Response Format Compliance (does it match expected JSON structure?)
    - Response Time
    - Success Rate
    
    ✅ ADVANCED METRICS:
    - User Intent Matching (does query answer the question?)
    """
    
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        output_dir: str = "/opt/airflow/outputs/evaluation",
        service_account_path: Optional[str] = None
    ):
        """
        Initialize model evaluator
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            output_dir: Directory to save evaluation reports
            service_account_path: Optional path to service account JSON
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(self.__class__.__name__)
        
        # Setup GCP credentials (optional - logs warning if not available)
        from utils import setup_gcp_credentials
        try:
            setup_gcp_credentials(service_account_path, self.logger)
        except Exception as e:
            self.logger.warning(f"GCP credentials not configured: {e}")
        
        # Initialize BigQuery client (optional - only needed for execution testing)
        try:
            self.bq_client = bigquery.Client(project=project_id)
            self.logger.info("✓ BigQuery client initialized")
        except Exception as e:
            self.logger.warning(f"BigQuery client not available (this is OK for testing): {e}")
            self.bq_client = None
        
        # User intent keywords for matching
        self.intent_keywords = self._build_intent_keywords()
    
    def _build_intent_keywords(self) -> Dict[str, List[str]]:
        """
        Build keyword patterns for intent matching
        
        Returns:
            Dictionary mapping intent types to keyword patterns
        """
        return {
            'aggregation': [
                'sum', 'total', 'count', 'average', 'avg', 'max', 'min',
                'how many', 'how much', 'total number'
            ],
            'filtering': [
                'where', 'filter', 'specific', 'only', 'show me',
                'for', 'in', 'with', 'having'
            ],
            'sorting': [
                'top', 'bottom', 'best', 'worst', 'highest', 'lowest',
                'rank', 'order', 'sort'
            ],
            'grouping': [
                'by', 'per', 'each', 'group', 'breakdown', 'category',
                'segment', 'across'
            ],
            'temporal': [
                'trend', 'over time', 'daily', 'monthly', 'yearly',
                'date', 'time', 'when', 'period'
            ],
            'comparison': [
                'compare', 'versus', 'vs', 'difference', 'between',
                'against', 'relative to'
            ],
            'listing': [
                'list', 'show', 'display', 'all', 'details', 'records'
            ]
        }
    def _validate_against_ground_truth(self, sql_query, expected_criteria):
        """Compare generated SQL against known correct patterns"""
        score = 0
    
    # Check if required SQL elements are present
        for required in expected_criteria.get('expected_sql_contains', []):
            if required.upper() in sql_query.upper():
                score += 1
    
        # Check result count matches
        # Check answer is in expected range
        return score / len(expected_criteria['expected_sql_contains'])
    
    def evaluate_model_responses(
        self,
        model_name: str,
        responses: List[Dict],
        test_dataframe: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Evaluate all responses from a single model
        
        Args:
            model_name: Name of the model being evaluated
            responses: List of response dictionaries with:
                - query_number, user_query, sql_query, visualization, 
                  explanation, response_time, status
            test_dataframe: Optional DataFrame for SQL execution testing
            
        Returns:
            Comprehensive evaluation report dictionary
        """
        self.logger.info(f"Evaluating model: {model_name}")
        self.logger.info(f"Total responses: {len(responses)}")
        
        # Initialize metrics
        metrics = {
            "model_name": model_name,
            "total_queries": len(responses),
            "evaluation_timestamp": datetime.now().isoformat(),
            
            # Basic metrics
            "syntax_valid": 0,
            "executable": 0,
            "format_compliant": 0,
            "avg_response_time": 0,
            "success_rate": 0,
            
            # Advanced metrics
            "intent_matched": 0,
            
            "detailed_results": []
        }
        
        total_response_time = 0
        
        # Evaluate each response
        for response in responses:
            if response.get('status') != 'success':
                # Skip failed responses
                metrics['detailed_results'].append({
                    "query_number": response.get('query_number'),
                    "user_query": response.get('user_query'),
                    "status": "failed",
                    "error": response.get('error', 'Unknown error')
                })
                continue
            
            # Evaluate this response
            result = self._evaluate_single_response(
                response,
                test_dataframe
            )
            
            metrics['detailed_results'].append(result)
            
            # Update aggregate metrics - Basic
            if result['syntax_valid']:
                metrics['syntax_valid'] += 1
            if result['executable']:
                metrics['executable'] += 1
            if result['format_compliant']:
                metrics['format_compliant'] += 1
            
            # Update aggregate metrics - Advanced
            if result.get('intent_matched', False):
                metrics['intent_matched'] += 1
            
            total_response_time += response.get('response_time', 0)
        
        # Calculate percentages
        successful_responses = len([r for r in responses if r.get('status') == 'success'])
        
        if successful_responses > 0:
            # Basic rates
            metrics['syntax_validity_rate'] = (metrics['syntax_valid'] / successful_responses) * 100
            metrics['executability_rate'] = (metrics['executable'] / successful_responses) * 100
            metrics['format_compliance_rate'] = (metrics['format_compliant'] / successful_responses) * 100
            metrics['avg_response_time'] = total_response_time / successful_responses
            
            # Advanced rates
            metrics['intent_matching_rate'] = (metrics['intent_matched'] / successful_responses) * 100
        
        metrics['success_rate'] = (successful_responses / len(responses)) * 100
        
        # Calculate overall score
        metrics['overall_score'] = self._calculate_overall_score(metrics)
        
        self.logger.info(f"✓ Evaluation complete for {model_name}")
        self.logger.info(f"  Overall Score: {metrics['overall_score']:.2f}/100")
        self.logger.info(f"  Success Rate: {metrics['success_rate']:.2f}%")
        self.logger.info(f"  Syntax Valid: {metrics.get('syntax_validity_rate', 0):.2f}%")
        self.logger.info(f"  Intent Matched: {metrics.get('intent_matching_rate', 0):.2f}%")
        
        return metrics
    
    def _evaluate_single_response(
        self,
        response: Dict,
        test_dataframe: Optional[pd.DataFrame]
    ) -> Dict:
        """Evaluate a single response with all metrics"""
        result = {
            "query_number": response.get('query_number'),
            "user_query": response.get('user_query'),
            
            # Basic metrics
            "syntax_valid": False,
            "executable": False,
            "format_compliant": False,
            "sql_length": 0,
            "explanation_length": 0,
            "visualization_type": None,
            
            # Advanced metrics
            "intent_matched": False,
            
            "errors": [],
            "intent_analysis": {}
        }
        
        sql_query = response.get('sql_query', '')
        visualization = response.get('visualization', {})
        explanation = response.get('explanation', '')
        user_query = response.get('user_query', '')
        
        # ========================================
        # BASIC METRICS
        # ========================================
        
        # 1. Check SQL syntax validity
        result['syntax_valid'], syntax_error = self._validate_sql_syntax(sql_query)
        if not result['syntax_valid']:
            result['errors'].append(f"Syntax error: {syntax_error}")
        
        # 2. Check SQL executability
        if test_dataframe is not None and result['syntax_valid']:
            result['executable'], exec_error = self._test_sql_execution(
                sql_query,
                test_dataframe
            )
            if not result['executable']:
                result['errors'].append(f"Execution error: {exec_error}")
        
        # 3. Check format compliance
        result['format_compliant'] = self._validate_response_format(
            sql_query,
            visualization,
            explanation
        )
        if not result['format_compliant']:
            result['errors'].append("Response format non-compliant")
        
        # 4. Extract metadata
        result['sql_length'] = len(sql_query)
        result['explanation_length'] = len(explanation)
        result['visualization_type'] = visualization.get('type', 'unknown')
        
        # ========================================
        # ADVANCED METRICS
        # ========================================
        
        # 5. User Intent Matching
        result['intent_matched'], result['intent_analysis'] = self._check_intent_matching(
            user_query,
            sql_query
        )
        
        return result
    
    def _validate_sql_syntax(self, sql: str) -> Tuple[bool, str]:
        """Validate SQL syntax using DuckDB"""
        try:
            conn = duckdb.connect(database=':memory:')
            conn.execute("CREATE TABLE dataset (id INTEGER)")
            conn.execute(f"EXPLAIN {sql}")
            conn.close()
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def _test_sql_execution(
        self,
        sql: str,
        df: pd.DataFrame
    ) -> Tuple[bool, str]:
        """Test if SQL can execute on sample data"""
        try:
            conn = duckdb.connect(database=':memory:')
            conn.register('dataset', df)
            
            # Add LIMIT for safety
            if 'LIMIT' not in sql.upper():
                sql = sql.rstrip(';') + ' LIMIT 100;'
            
            result = conn.execute(sql).df()
            conn.close()
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def _validate_response_format(
        self,
        sql: str,
        visualization: Dict,
        explanation: str
    ) -> bool:
        """Check if response has all required components"""
        # Check SQL exists and is not empty
        if not sql or len(sql.strip()) < 10:
            return False
        
        # Check visualization has required fields
        if not isinstance(visualization, dict):
            return False
        if 'type' not in visualization or 'title' not in visualization:
            return False
        
        valid_viz_types = ['bar_chart', 'line_chart', 'pie_chart', 'scatter_plot', 'table']
        if visualization.get('type') not in valid_viz_types:
            return False
        
        # Check explanation exists
        if not explanation or len(explanation.strip()) < 10:
            return False
        
        return True
    
    def _check_intent_matching(
        self,
        user_query: str,
        sql_query: str
    ) -> Tuple[bool, Dict]:
        """
        Check if SQL query matches user intent
        
        Args:
            user_query: Original user question
            sql_query: Generated SQL query
            
        Returns:
            Tuple of (intent_matched: bool, analysis: dict)
        """
        user_query_lower = user_query.lower()
        sql_query_upper = sql_query.upper()
        
        analysis = {
            "user_intents_detected": [],
            "sql_patterns_found": [],
            "matched_intents": [],
            "missing_intents": []
        }
        
        # Detect intents in user query
        for intent_type, keywords in self.intent_keywords.items():
            if any(keyword in user_query_lower for keyword in keywords):
                analysis["user_intents_detected"].append(intent_type)
        
        # Check SQL patterns
        sql_patterns = {
            'aggregation': bool(re.search(r'\b(SUM|AVG|COUNT|MAX|MIN)\s*\(', sql_query_upper)),
            'filtering': 'WHERE' in sql_query_upper,
            'sorting': 'ORDER BY' in sql_query_upper,
            'grouping': 'GROUP BY' in sql_query_upper,
            'temporal': bool(re.search(r'\b(DATE|TIME|YEAR|MONTH|DAY)\b', sql_query_upper)),
            'comparison': bool(re.search(r'\b(CASE|WHEN|THEN)\b', sql_query_upper)),
            'listing': 'SELECT' in sql_query_upper and 'GROUP BY' not in sql_query_upper
        }
        
        for pattern, found in sql_patterns.items():
            if found:
                analysis["sql_patterns_found"].append(pattern)
        
        # Match intents with SQL patterns
        for intent in analysis["user_intents_detected"]:
            if intent in analysis["sql_patterns_found"]:
                analysis["matched_intents"].append(intent)
            else:
                analysis["missing_intents"].append(intent)
        
        # Intent is matched if:
        # 1. At least one intent detected in user query
        # 2. All detected intents are matched in SQL
        intent_matched = (
            len(analysis["user_intents_detected"]) > 0 and
            len(analysis["missing_intents"]) == 0
        )
        
        return intent_matched, analysis
    
    def _calculate_overall_score(self, metrics: Dict) -> float:
        """
        Calculate weighted overall score
        
        WEIGHTS:
        - Success Rate: 20%
        - Syntax Validity: 20%
        - Executability: 20%
        - Format Compliance: 15%
        - Intent Matching: 25%
        """
        weights = {
            'success_rate': 0.20,
            'syntax_validity_rate': 0.20,
            'executability_rate': 0.20,
            'format_compliance_rate': 0.15,
            'intent_matching_rate': 0.25,
        }
        
        score = 0.0
        
        for metric, weight in weights.items():
            value = metrics.get(metric, 0)
            score += value * weight
        
        return score
    #Sensitivity Analysis
    #sensitivity means: How does changing prompt structure affect output?
                        #does temperature (0.1 vs 0.9) impact SQL quality?
                        #does context length affect accuracy?
                        #how few-shot examples influence results?

    def perform_sensitivity_analysis(self, model_name, test_queries):
        """
        Sensitivity analysis for LLM model
        Tests how model responds to variations in:
        - Temperature (0.2, 0.4, 0.6, 0.8)
        - Prompt structure (minimal, standard, detailed)
        - Context length (100, 500, 1000 tokens)
        - Few-shot examples (0, 3, 5, 10 examples)
        """
        sensitivity_report = {
            'temperature_sensitivity': {},
            'prompt_sensitivity': {},
            'context_sensitivity': {},
            'example_sensitivity': {}
        }
        
        # Test 1: Temperature sensitivity
        for temp in [0.2, 0.4, 0.6, 0.8]:
            # Run same queries with different temperature
            # Measure consistency vs creativity
            pass
        
        # Test 2: Prompt structure sensitivity
        prompt_variations = ['minimal', 'standard', 'detailed']
        for variation in prompt_variations:
            # Test with different prompt styles
            pass
        
        # Test 3: Context length sensitivity
        # Test 4: Few-shot example sensitivity
        
        return sensitivity_report

    def save_evaluation_report(
        self,
        metrics: Dict,
        model_name: str
    ) -> str:
        """
        Save evaluation report to JSON file
        
        Args:
            metrics: Evaluation metrics dictionary
            model_name: Name of the model
            
        Returns:
            Path to saved report file
        """
        # Clean model name for filename
        clean_name = model_name.replace('/', '_').replace('.', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report_file = self.output_dir / f"{clean_name}_evaluation_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"✓ Saved evaluation report: {report_file}")
        
        return str(report_file)
    
    def generate_comparison_report(
        self,
        all_model_metrics: Dict[str, Dict]
    ) -> str:
        """
        Generate comparison report across all models
        
        Args:
            all_model_metrics: Dictionary mapping model names to their metrics
            
        Returns:
            Path to comparison report file
        """
        self.logger.info("Generating model comparison report...")
        
        comparison = {
            "comparison_timestamp": datetime.now().isoformat(),
            "models_evaluated": len(all_model_metrics),
            "summary": {},
            "detailed_comparison": []
        }
        
        # Create comparison table
        for model_name, metrics in all_model_metrics.items():
            comparison['detailed_comparison'].append({
                "model": model_name,
                "overall_score": metrics.get('overall_score', 0),
                
                # Basic metrics
                "success_rate": metrics.get('success_rate', 0),
                "syntax_validity": metrics.get('syntax_validity_rate', 0),
                "executability": metrics.get('executability_rate', 0),
                "format_compliance": metrics.get('format_compliance_rate', 0),
                "avg_response_time": metrics.get('avg_response_time', 0),
                
                # Advanced metrics
                "intent_matching": metrics.get('intent_matching_rate', 0)
            })
        
        # Sort by overall score
        comparison['detailed_comparison'].sort(
            key=lambda x: x['overall_score'],
            reverse=True
        )
        
        # Identify best model
        if comparison['detailed_comparison']:
            best_model = comparison['detailed_comparison'][0]
            comparison['summary']['best_model'] = best_model['model']
            comparison['summary']['best_score'] = best_model['overall_score']
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"model_comparison_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        self.logger.info(f"✓ Saved comparison report: {report_file}")
        self.logger.info(f"  Best Model: {comparison['summary'].get('best_model')}")
        self.logger.info(f"  Best Score: {comparison['summary'].get('best_score', 0):.2f}/100")
        
        return str(report_file)


# ========================================
# STANDALONE TESTING
# ========================================

if __name__ == "__main__":
    """
    Test script with enhanced metrics
    Usage: python model_evaluator.py
    """
    
    # Mock responses for testing
    mock_responses = [
        {
            "query_number": 1,
            "user_query": "What are the top 5 products by sales?",
            "sql_query": "SELECT product_name, SUM(sales) as total FROM dataset GROUP BY product_name ORDER BY total DESC LIMIT 5",
            "visualization": {"type": "bar_chart", "title": "Top Products"},
            "explanation": "Shows the top 5 products ranked by total sales volume",
            "response_time": 1.5,
            "status": "success"
        },
        {
            "query_number": 2,
            "user_query": "Show me sales trend over time",
            "sql_query": "SELECT order_date, SUM(sales) FROM dataset GROUP BY order_date ORDER BY order_date",
            "visualization": {"type": "line_chart", "title": "Sales Trend"},
            "explanation": "Time series analysis showing daily sales trends",
            "response_time": 1.2,
            "status": "success"
        }
    ]
    
    print("\n" + "=" * 60)
    print("TESTING MODEL EVALUATOR (ENHANCED)")
    print("=" * 60 + "\n")
    
    try:
        # Initialize evaluator
        print("1. Initializing evaluator...")
        evaluator = ModelEvaluator(
            project_id="datacraft-data-pipeline",
            dataset_id="datacraft_ml",
            output_dir="/tmp/evaluation"
        )
        print("   ✓ Evaluator initialized (BigQuery optional for testing)\n")
        
        # Evaluate mock responses (without SQL execution testing)
        print("2. Evaluating mock responses with ENHANCED metrics...")
        metrics = evaluator.evaluate_model_responses(
            model_name="gemini-test",
            responses=mock_responses,
            test_dataframe=None  # Skip SQL execution tests
        )
        
        print(f"\n3. Evaluation Results:")
        print(f"   📊 BASIC METRICS:")
        print(f"   Overall Score: {metrics['overall_score']:.2f}/100")
        print(f"   Success Rate: {metrics['success_rate']:.2f}%")
        print(f"   Syntax Valid: {metrics.get('syntax_validity_rate', 0):.2f}%")
        print(f"   Format Compliant: {metrics.get('format_compliance_rate', 0):.2f}%")
        
        print(f"\n   🎯 ADVANCED METRICS:")
        print(f"   Intent Matching: {metrics.get('intent_matching_rate', 0):.2f}%")
        
        # Show detailed results for first query
        if metrics['detailed_results']:
            first_result = metrics['detailed_results'][0]
            print(f"\n4. Sample Detailed Result (Query 1):")
            print(f"   Intent Analysis: {first_result.get('intent_analysis', {})}")
        
        # Save report
        print("\n5. Saving evaluation report...")
        report_file = evaluator.save_evaluation_report(metrics, "gemini-test")
        print(f"   ✓ Report saved: {report_file}")
        
        print("\n" + "=" * 60)
        print("✓ MODEL EVALUATOR TEST PASSED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)