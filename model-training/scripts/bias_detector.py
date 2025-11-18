# scripts/bias_detector.py
"""
Bias Detection for Query & Visualization Models
Detects potential biases in LLM-generated SQL queries and visualizations
Focus areas: query complexity fairness, visualization type distribution, demographic representation
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging
from datetime import datetime
from collections import Counter
import sys
import re

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging


class BiasDetector:
    """
    Detect biases in model-generated queries and visualizations
    
    Bias Categories Evaluated:
    1. Query Generation Bias:
       - Does the model favor certain types of queries?
       - Are aggregations consistently used over detailed queries?
       
    2. Visualization Selection Bias:
       - Over-reliance on specific chart types
       - Underutilization of certain visualization types
       
    3. Sentiment Bias (in explanations):
       - Overly positive or negative language
       - Inconsistent confidence levels
       
    4. Column Selection Bias:
       - Does the model favor certain columns?
       - Are categorical vs numeric columns treated fairly?
    """
    
    def __init__(self, output_dir: str = "/opt/airflow/outputs/bias"):
        """
        Initialize bias detector
        
        Args:
            output_dir: Directory to save bias reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(self.__class__.__name__)
        
        # Bias thresholds
        self.THRESHOLDS = {
            'visualization_imbalance': 0.60,  # >60% usage is biased
            'query_pattern_imbalance': 0.70,
            'column_usage_imbalance': 0.50
        }
    
    def detect_all_biases(
        self,
        model_name: str,
        responses: List[Dict],
        dataset_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Comprehensive bias detection across all response dimensions
        
        Args:
            model_name: Name of the model being evaluated
            responses: List of model responses
            dataset_metadata: Optional metadata about the dataset structure
            
        Returns:
            Comprehensive bias report dictionary
        """
        self.logger.info(f"Running bias detection for: {model_name}")
        self.logger.info(f"Analyzing {len(responses)} responses")
        
        # Initialize bias report
        bias_report = {
            "model_name": model_name,
            "total_responses": len(responses),
            "analysis_timestamp": datetime.now().isoformat(),
            "bias_detected": False,
            "bias_score": 0.0,  # 0-100, lower is better
            "biases": []
        }
        
        # Filter successful responses
        successful_responses = [
            r for r in responses 
            if r.get('status') == 'success'
        ]
        
        if len(successful_responses) < 2:
            self.logger.warning("Not enough successful responses for bias detection")
            return bias_report
        
        # 1. Visualization Type Bias
        viz_bias = self._detect_visualization_bias(successful_responses)
        if viz_bias['biased']:
            bias_report['biases'].append(viz_bias)
            bias_report['bias_detected'] = True
        
        # 2. Query Pattern Bias
        query_bias = self._detect_query_pattern_bias(successful_responses)
        if query_bias['biased']:
            bias_report['biases'].append(query_bias)
            bias_report['bias_detected'] = True
        
        # 3. Column Usage Bias
        if dataset_metadata:
            column_bias = self._detect_column_usage_bias(
                successful_responses,
                dataset_metadata
            )
            if column_bias['biased']:
                bias_report['biases'].append(column_bias)
                bias_report['bias_detected'] = True

        # 4. Demographic / Slice Bias (
            demographic_bias = self._detect_demographic_bias(
                responses,          # use ALL responses (need failures too)
                dataset_metadata
            )
            if demographic_bias['biased']:
                bias_report['biases'].append(demographic_bias)
                bias_report['bias_detected'] = True
        
        # 5. Sentiment Bias
        sentiment_bias = self._detect_sentiment_bias(successful_responses)
        if sentiment_bias['biased']:
            bias_report['biases'].append(sentiment_bias)
            bias_report['bias_detected'] = True
        
        # Calculate overall bias score
        bias_report['bias_score'] = self._calculate_bias_score(bias_report['biases'])
        
        # Severity rating
        bias_report['severity'] = self._get_severity_rating(bias_report['bias_score'])
        
        self.logger.info(f"✓ Bias detection complete")
        self.logger.info(f"  Bias Detected: {bias_report['bias_detected']}")
        self.logger.info(f"  Bias Score: {bias_report['bias_score']:.2f}/100")
        self.logger.info(f"  Severity: {bias_report['severity']}")
        
        return bias_report
    
    def _detect_visualization_bias(self, responses: List[Dict]) -> Dict:
        """
        Detect if model over-relies on certain visualization types
        """
        viz_types = [
            r.get('visualization', {}).get('type', 'unknown')
            for r in responses
        ]
        
        # Count occurrences
        viz_counts = Counter(viz_types)
        total = len(viz_types)
        
        # Calculate distribution
        distribution = {
            viz_type: (count / total) * 100
            for viz_type, count in viz_counts.items()
        }
        
        # Check for imbalance
        max_usage = max(distribution.values()) if distribution else 0
        biased = max_usage > (self.THRESHOLDS['visualization_imbalance'] * 100)
        
        bias_info = {
            "category": "Visualization Selection Bias",
            "biased": biased,
            "severity": "HIGH" if max_usage > 80 else "MEDIUM" if biased else "LOW",
            "details": {
                "distribution": distribution,
                "max_usage_pct": max_usage,
                "threshold_pct": self.THRESHOLDS['visualization_imbalance'] * 100
            },
            "description": f"Model uses {list(distribution.keys())[0] if distribution else 'unknown'} " +
                          f"in {max_usage:.1f}% of cases"
        }
        
        return bias_info
    
    def _detect_query_pattern_bias(self, responses: List[Dict]) -> Dict:
        """
        Detect if model favors certain SQL patterns (aggregation, filtering, etc.)
        """
        sql_queries = [r.get('sql_query', '') for r in responses]
        
        # Analyze query patterns
        patterns = {
            'has_aggregation': 0,
            'has_group_by': 0,
            'has_where': 0,
            'has_join': 0,
            'has_order_by': 0,
            'simple_select': 0
        }
        
        for sql in sql_queries:
            sql_upper = sql.upper()
            
            if any(agg in sql_upper for agg in ['SUM(', 'AVG(', 'COUNT(', 'MAX(', 'MIN(']):
                patterns['has_aggregation'] += 1
            if 'GROUP BY' in sql_upper:
                patterns['has_group_by'] += 1
            if 'WHERE' in sql_upper:
                patterns['has_where'] += 1
            if 'JOIN' in sql_upper:
                patterns['has_join'] += 1
            if 'ORDER BY' in sql_upper:
                patterns['has_order_by'] += 1
            if sql_upper.count('SELECT') == 1 and 'GROUP BY' not in sql_upper:
                patterns['simple_select'] += 1
        
        # Calculate percentages
        total = len(sql_queries)
        pattern_pct = {
            k: (v / total) * 100
            for k, v in patterns.items()
        }
        
        # Check for over-reliance on aggregations
        agg_bias = pattern_pct.get('has_aggregation', 0) > (self.THRESHOLDS['query_pattern_imbalance'] * 100)
        
        bias_info = {
            "category": "Query Pattern Bias",
            "biased": agg_bias,
            "severity": "MEDIUM" if agg_bias else "LOW",
            "details": {
                "pattern_distribution": pattern_pct,
                "threshold_pct": self.THRESHOLDS['query_pattern_imbalance'] * 100
            },
            "description": f"Model uses aggregations in {pattern_pct.get('has_aggregation', 0):.1f}% of queries"
        }
        
        return bias_info
    
    def _detect_column_usage_bias(
        self,
        responses: List[Dict],
        dataset_metadata: Dict
    ) -> Dict:
        """
        Detect if model favors certain columns over others
        """
        # Extract all columns from dataset
        all_columns = set()
        if 'columns' in dataset_metadata:
            all_columns = {col['name'] for col in dataset_metadata['columns']}
        
        # Count column mentions in SQL queries
        column_usage = Counter()
        
        for response in responses:
            sql = response.get('sql_query', '')
            # Simple extraction: look for column names in SQL
            for col in all_columns:
                if col in sql:
                    column_usage[col] += 1
        
        # Calculate usage distribution
        total_queries = len(responses)
        usage_pct = {
            col: (count / total_queries) * 100
            for col, count in column_usage.items()
        }
        
        # Check for imbalance
        max_usage = max(usage_pct.values()) if usage_pct else 0
        min_usage = min(usage_pct.values()) if usage_pct else 0
        usage_gap = max_usage - min_usage
        
        biased = usage_gap > (self.THRESHOLDS['column_usage_imbalance'] * 100)
        
        bias_info = {
            "category": "Column Usage Bias",
            "biased": biased,
            "severity": "MEDIUM" if biased else "LOW",
            "details": {
                "column_usage_pct": usage_pct,
                "max_usage": max_usage,
                "min_usage": min_usage,
                "usage_gap": usage_gap,
                "threshold_gap": self.THRESHOLDS['column_usage_imbalance'] * 100
            },
            "description": f"Usage gap between most and least used columns: {usage_gap:.1f}%"
        }
        
        return bias_info
    
    def _detect_sentiment_bias(self, responses: List[Dict]) -> Dict:
        """
        Detect sentiment bias in generated explanations
        """
        explanations = [r.get('explanation', '') for r in responses]
        
        # Simple sentiment analysis using keyword counting
        positive_words = ['excellent', 'great', 'best', 'top', 'high', 'good', 'optimal']
        negative_words = ['poor', 'low', 'worst', 'bad', 'minimal', 'least']
        neutral_words = ['shows', 'displays', 'indicates', 'presents', 'analyzes']
        
        sentiment_counts = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        for exp in explanations:
            exp_lower = exp.lower()
            
            pos_count = sum(1 for word in positive_words if word in exp_lower)
            neg_count = sum(1 for word in negative_words if word in exp_lower)
            neu_count = sum(1 for word in neutral_words if word in exp_lower)
            
            if pos_count > neg_count and pos_count > neu_count:
                sentiment_counts['positive'] += 1
            elif neg_count > pos_count and neg_count > neu_count:
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1
        
        # Calculate distribution
        total = len(explanations)
        sentiment_pct = {
            k: (v / total) * 100
            for k, v in sentiment_counts.items()
        }
        
        # Check for extreme bias
        max_sentiment = max(sentiment_pct.values())
        biased = max_sentiment > 70
        
        bias_info = {
            "category": "Sentiment Bias",
            "biased": biased,
            "severity": "LOW",
            "details": {
                "sentiment_distribution": sentiment_pct,
                "dominant_sentiment": max(sentiment_pct, key=sentiment_pct.get)
            },
            "description": f"Sentiment distribution: {sentiment_pct}"
        }
        
        return bias_info
    
    def _calculate_bias_score(self, biases: List[Dict]) -> float:
        """
        Calculate overall bias score (0-100, lower is better)
        
        Weights by severity:
        - HIGH: 40 points
        - MEDIUM: 25 points
        - LOW: 10 points
        """
        severity_weights = {
            'HIGH': 40,
            'MEDIUM': 25,
            'LOW': 10
        }
        
        score = 0.0
        for bias in biases:
            if bias.get('biased', False):
                severity = bias.get('severity', 'LOW')
                score += severity_weights.get(severity, 10)
        
        return min(score, 100)  # Cap at 100
    
    def _get_severity_rating(self, bias_score: float) -> str:
        """Convert bias score to severity rating"""
        if bias_score >= 60:
            return "HIGH"
        elif bias_score >= 30:
            return "MEDIUM"
        else:
            return "LOW"
    
    def save_bias_report(
        self,
        bias_report: Dict,
        model_name: str
    ) -> str:
        """
        Save bias report to JSON file
        
        Args:
            bias_report: Bias report dictionary
            model_name: Name of the model
            
        Returns:
            Path to saved report file
        """
        # Clean model name for filename
        clean_name = model_name.replace('/', '_').replace('.', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report_file = self.output_dir / f"{clean_name}_bias_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(bias_report, f, indent=2)
        
        self.logger.info(f"✓ Saved bias report: {report_file}")
        
        return str(report_file)
    
    def generate_bias_comparison(
        self,
        all_model_bias_reports: Dict[str, Dict]
    ) -> str:
        """
        Generate comparison of bias across all models
        
        Args:
            all_model_bias_reports: Dictionary mapping model names to bias reports
            
        Returns:
            Path to comparison report file
        """
        self.logger.info("Generating bias comparison report...")
        
        comparison = {
            "comparison_timestamp": datetime.now().isoformat(),
            "models_analyzed": len(all_model_bias_reports),
            "bias_comparison": []
        }
        
        for model_name, report in all_model_bias_reports.items():
            comparison['bias_comparison'].append({
                "model": model_name,
                "bias_detected": report.get('bias_detected', False),
                "bias_score": report.get('bias_score', 0),
                "severity": report.get('severity', 'UNKNOWN'),
                "num_biases": len(report.get('biases', []))
            })
        
        # Sort by bias score (ascending - lower is better)
        comparison['bias_comparison'].sort(
            key=lambda x: x['bias_score']
        )
        
        # Identify least biased model
        if comparison['bias_comparison']:
            best_model = comparison['bias_comparison'][0]
            comparison['least_biased_model'] = best_model['model']
            comparison['lowest_bias_score'] = best_model['bias_score']
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"bias_comparison_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        self.logger.info(f"✓ Saved bias comparison: {report_file}")
        self.logger.info(f"  Least Biased Model: {comparison.get('least_biased_model')}")
        self.logger.info(f"  Lowest Bias Score: {comparison.get('lowest_bias_score', 0):.2f}/100")
        
        return str(report_file)


# ========================================
# STANDALONE TESTING
# ========================================

if __name__ == "__main__":
    """
    Test script
    Usage: python bias_detector.py
    """
    
    # Mock responses for testing
    mock_responses = [
        {
            "query_number": 1,
            "user_query": "What are top products?",
            "sql_query": "SELECT product_name, SUM(sales) as total FROM dataset GROUP BY product_name ORDER BY total DESC",
            "visualization": {"type": "bar_chart", "title": "Top Products"},
            "explanation": "Shows excellent performance of top products",
            "status": "success"
        },
        {
            "query_number": 2,
            "user_query": "Sales trend",
            "sql_query": "SELECT order_date, SUM(sales) FROM dataset GROUP BY order_date",
            "visualization": {"type": "bar_chart", "title": "Sales"},
            "explanation": "Displays great sales trend over time",
            "status": "success"
        },
        {
            "query_number": 3,
            "user_query": "Product categories",
            "sql_query": "SELECT category, COUNT(*) FROM dataset GROUP BY category",
            "visualization": {"type": "bar_chart", "title": "Categories"},
            "explanation": "Presents optimal category distribution",
            "status": "success"
        }
    ]
    
    print("\n" + "=" * 60)
    print("TESTING BIAS DETECTOR")
    print("=" * 60 + "\n")
    
    try:
        # Initialize detector
        detector = BiasDetector(output_dir="/tmp/bias")
        
        # Run bias detection
        print("1. Running bias detection...")
        bias_report = detector.detect_all_biases(
            model_name="gemini-test",
            responses=mock_responses
        )
        
        print(f"\n2. Bias Detection Results:")
        print(f"   Bias Detected: {bias_report['bias_detected']}")
        print(f"   Bias Score: {bias_report['bias_score']:.2f}/100")
        print(f"   Severity: {bias_report['severity']}")
        print(f"   Number of Biases: {len(bias_report['biases'])}")
        
        # Save report
        print("\n3. Saving bias report...")
        report_file = detector.save_bias_report(bias_report, "gemini-test")
        print(f"   Report saved: {report_file}")
        
        print("\n" + "=" * 60)
        print("✓ BIAS DETECTOR TEST PASSED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    