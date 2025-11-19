# scripts/sensitivity_analysis.py
"""
Sensitivity Analysis for Model Performance
Analyzes how model behavior changes with different conditions
"""

import json
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging


class SensitivityAnalyzer:
    """
    Analyze model sensitivity to different factors
    
    Tests:
    - Query complexity impact on performance
    - Response consistency across similar queries
    - Performance stability
    """
    
    def __init__(self, project_id: str, output_dir: str = "/opt/airflow/outputs/sensitivity"):
        """
        Initialize sensitivity analyzer
        
        Args:
            project_id: GCP project ID
            output_dir: Directory to save sensitivity reports
        """
        self.project_id = project_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(self.__class__.__name__)
    
    def analyze_model_sensitivity(
        self,
        model_name: str,
        responses: List[Dict],
        metadata: Dict
    ) -> Dict:
        """
        Comprehensive sensitivity analysis
        
        Args:
            model_name: Name of the model
            responses: List of model responses
            metadata: Dataset metadata
            
        Returns:
            Sensitivity analysis report
        """
        self.logger.info(f"Running sensitivity analysis for: {model_name}")
        self.logger.info(f"Analyzing {len(responses)} responses")
        
        sensitivity_report = {
            "model_name": model_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_responses": len(responses),
            "analyses": {}
        }
        
        successful = [r for r in responses if r.get('status') == 'success']
        
        if len(successful) < 5:
            self.logger.warning("Not enough responses for sensitivity analysis")
            return sensitivity_report
        
        # Analysis 1: Performance by query type
        query_type_analysis = self._analyze_by_query_type(successful)
        sensitivity_report['analyses']['query_type_sensitivity'] = query_type_analysis
        
        # Analysis 2: Response time stability
        time_stability = self._analyze_response_time_stability(successful)
        sensitivity_report['analyses']['response_time_stability'] = time_stability
        
        # Analysis 3: Output format consistency
        format_consistency = self._analyze_output_consistency(successful)
        sensitivity_report['analyses']['output_consistency'] = format_consistency
        
        self.logger.info(f"✓ Sensitivity analysis complete")
        
        return sensitivity_report
    
    def _analyze_by_query_type(self, responses: List[Dict]) -> Dict:
        """Analyze performance across different query types"""
        query_types = {
            'aggregation': [],
            'filtering': [],
            'sorting': [],
            'listing': []
        }
        
        for response in responses:
            user_query = response.get('user_query', '').lower()
            
            if any(kw in user_query for kw in ['total', 'sum', 'average', 'count']):
                query_types['aggregation'].append(response)
            elif any(kw in user_query for kw in ['top', 'best', 'highest']):
                query_types['sorting'].append(response)
            elif any(kw in user_query for kw in ['list', 'show all', 'display']):
                query_types['listing'].append(response)
            else:
                query_types['filtering'].append(response)
        
        # Calculate performance by type
        type_performance = {}
        for qtype, qlist in query_types.items():
            if qlist:
                avg_time = np.mean([q.get('response_time', 0) for q in qlist])
                type_performance[qtype] = {
                    "count": len(qlist),
                    "avg_response_time": float(avg_time)
                }
        
        return {
            "query_type_distribution": {k: len(v) for k, v in query_types.items()},
            "performance_by_type": type_performance,
            "most_common_type": max(query_types, key=lambda k: len(query_types[k]))
        }
    
    def _analyze_response_time_stability(self, responses: List[Dict]) -> Dict:
        """Analyze response time variance"""
        times = [r.get('response_time', 0) for r in responses if r.get('response_time', 0) > 0]
        
        if len(times) < 2:
            return {"status": "insufficient_data"}
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = (std_time / mean_time) if mean_time > 0 else 0
        
        return {
            "mean_time": float(mean_time),
            "std_time": float(std_time),
            "coefficient_of_variation": float(cv),
            "stability_rating": "stable" if cv < 0.3 else "moderate" if cv < 0.6 else "unstable",
            "time_range": {
                "min": float(np.min(times)),
                "max": float(np.max(times))
            }
        }
    
    def _analyze_output_consistency(self, responses: List[Dict]) -> Dict:
        """Analyze consistency of output formats"""
        viz_types = [r.get('visualization', {}).get('type') for r in responses]
        sql_lengths = [len(r.get('sql_query', '')) for r in responses]
        
        viz_distribution = dict(Counter(viz_types))
        
        return {
            "visualization_variety": len(viz_distribution),
            "sql_length_variance": float(np.std(sql_lengths)),
            "avg_sql_length": float(np.mean(sql_lengths)),
            "consistency_score": 100 - (float(np.std(sql_lengths)) / float(np.mean(sql_lengths)) * 100)
        }
    
    def save_sensitivity_report(
        self,
        sensitivity_report: Dict,
        model_name: str
    ) -> str:
        """Save sensitivity report to JSON"""
        clean_name = model_name.replace('/', '_').replace('.', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report_file = self.output_dir / f"{clean_name}_sensitivity_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(sensitivity_report, f, indent=2)
        
        self.logger.info(f"✓ Saved sensitivity report: {report_file}")
        
        return str(report_file)


# ========================================
# STANDALONE TESTING
# ========================================

if __name__ == "__main__":
    """Test script"""
    
    print("\n" + "=" * 60)
    print("TESTING SENSITIVITY ANALYZER")
    print("=" * 60 + "\n")
    
    # Mock responses
    mock_responses = [
        {
            "query_number": 1,
            "user_query": "What are total sales?",
            "sql_query": "SELECT SUM(sales) FROM dataset",
            "visualization": {"type": "bar_chart"},
            "response_time": 1.5,
            "status": "success"
        },
        {
            "query_number": 2,
            "user_query": "Show me top products",
            "sql_query": "SELECT product, SUM(sales) FROM dataset GROUP BY product ORDER BY sales DESC",
            "visualization": {"type": "bar_chart"},
            "response_time": 2.1,
            "status": "success"
        },
        {
            "query_number": 3,
            "user_query": "List all orders",
            "sql_query": "SELECT * FROM dataset LIMIT 100",
            "visualization": {"type": "table"},
            "response_time": 1.8,
            "status": "success"
        }
    ]
    
    try:
        analyzer = SensitivityAnalyzer(
            project_id="datacraft-data-pipeline",
            output_dir="/tmp/sensitivity"
        )
        
        print("✓ Analyzer initialized\n")
        
        print("Running sensitivity analysis...")
        report = analyzer.analyze_model_sensitivity(
            model_name="gemini-test",
            responses=mock_responses,
            metadata={}
        )
        
        print(f"\n📊 Sensitivity Results:")
        for test_name, test_result in report['analyses'].items():
            print(f"   {test_name}: {test_result}")
        
        # Save report
        report_file = analyzer.save_sensitivity_report(report, "gemini-test")
        print(f"\n✓ Report saved: {report_file}")
        
        print("\n" + "=" * 60)
        print("✓ TEST PASSED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)