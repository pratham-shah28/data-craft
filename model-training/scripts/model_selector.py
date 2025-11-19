# scripts/model_selector.py
"""
Model Selection Logic
Chooses the best model based on evaluation metrics and bias scores
"""

import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging


class ModelSelector:
    """
    Select the best model based on comprehensive evaluation and bias analysis
    
    Selection Criteria (weighted):
    - Overall Performance Score: 50%
    - Bias Score (inverted - lower bias is better): 30%
    - Response Time: 10%
    - Success Rate: 10%
    """
    
    def __init__(self, output_dir: str = "/opt/airflow/outputs/model-selection"):
        """
        Initialize model selector
        
        Args:
            output_dir: Directory to save selection reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(self.__class__.__name__)
        
        # Selection weights
        self.WEIGHTS = {
            'performance_score': 0.50,
            'bias_score': 0.30,  # Inverted: lower is better
            'response_time': 0.10,  # Inverted: lower is better
            'success_rate': 0.10
        }
    
    def select_best_model(
        self,
        evaluation_reports: Dict[str, Dict],
        bias_reports: Dict[str, Dict]
    ) -> Dict:
        """
        Select the best model from evaluation and bias reports
        
        Args:
            evaluation_reports: Dictionary mapping model names to evaluation metrics
            bias_reports: Dictionary mapping model names to bias reports
            
        Returns:
            Selection report with best model and detailed comparison
        """
        self.logger.info("Starting model selection process...")
        self.logger.info(f"Comparing {len(evaluation_reports)} models")
        
        # Ensure we have matching reports
        common_models = set(evaluation_reports.keys()) & set(bias_reports.keys())
        
        if not common_models:
            raise ValueError("No common models found in evaluation and bias reports!")
        
        # Calculate composite scores for each model
        model_scores = {}
        
        for model_name in common_models:
            eval_metrics = evaluation_reports[model_name]
            bias_metrics = bias_reports[model_name]
            
            composite_score = self._calculate_composite_score(
                eval_metrics,
                bias_metrics
            )
            
            model_scores[model_name] = composite_score
        
        # Sort by composite score (descending)
        ranked_models = sorted(
            model_scores.items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )
        
        # Select best model
        best_model_name, best_model_scores = ranked_models[0]
        
        # Create selection report
        selection_report = {
            "selection_timestamp": datetime.now().isoformat(),
            "models_compared": len(common_models),
            "best_model": {
                "name": best_model_name,
                "composite_score": best_model_scores['composite_score'],
                "performance_score": best_model_scores['performance_score'],
                "bias_score": best_model_scores['bias_score'],
                "response_time": best_model_scores['response_time'],
                "success_rate": best_model_scores['success_rate']
            },
            "all_models_ranking": [
                {
                    "rank": idx + 1,
                    "model": model_name,
                    "composite_score": scores['composite_score'],
                    "performance_score": scores['performance_score'],
                    "bias_score": scores['bias_score']
                }
                for idx, (model_name, scores) in enumerate(ranked_models)
            ],
            "selection_criteria": {
                "weights": self.WEIGHTS,
                "methodology": "Weighted composite score based on performance, bias, speed, and reliability"
            }
        }
        
        self.logger.info("✓ Model selection complete")
        self.logger.info(f"  Best Model: {best_model_name}")
        self.logger.info(f"  Composite Score: {best_model_scores['composite_score']:.2f}/100")
        self.logger.info(f"  Performance: {best_model_scores['performance_score']:.2f}/100")
        self.logger.info(f"  Bias Score: {best_model_scores['bias_score']:.2f}/100")
        
        return selection_report
    
    def _calculate_composite_score(
        self,
        eval_metrics: Dict,
        bias_metrics: Dict
    ) -> Dict:
        """
        Calculate weighted composite score
        
        Returns:
            Dictionary with component scores and composite score
        """
        # Extract metrics
        performance_score = eval_metrics.get('overall_score', 0)
        bias_score = bias_metrics.get('bias_score', 100)  # Lower is better
        response_time = eval_metrics.get('avg_response_time', 10)  # Lower is better
        success_rate = eval_metrics.get('success_rate', 0)
        
        # Normalize bias score (invert so higher is better)
        normalized_bias = 100 - bias_score
        
        # Normalize response time (assume max acceptable is 5 seconds)
        # Convert to 0-100 scale where faster is better
        normalized_time = max(0, 100 - (response_time / 5.0 * 100))
        
        # Calculate weighted composite score
        composite_score = (
            performance_score * self.WEIGHTS['performance_score'] +
            normalized_bias * self.WEIGHTS['bias_score'] +
            normalized_time * self.WEIGHTS['response_time'] +
            success_rate * self.WEIGHTS['success_rate']
        )
        
        return {
            "composite_score": composite_score,
            "performance_score": performance_score,
            "bias_score": bias_score,
            "response_time": response_time,
            "success_rate": success_rate,
            "normalized_bias": normalized_bias,
            "normalized_time": normalized_time
        }
    
    def save_selection_report(
        self,
        selection_report: Dict
    ) -> str:
        """
        Save selection report to JSON file
        
        Args:
            selection_report: Selection report dictionary
            
        Returns:
            Path to saved report file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"model_selection_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(selection_report, f, indent=2)
        
        self.logger.info(f"✓ Saved selection report: {report_file}")
        
        return str(report_file)
    
    def get_best_model_name(
        self,
        evaluation_reports: Dict[str, Dict],
        bias_reports: Dict[str, Dict]
    ) -> str:
        """
        Quick function to get just the best model name
        
        Args:
            evaluation_reports: Evaluation reports for all models
            bias_reports: Bias reports for all models
            
        Returns:
            Name of the best model
        """
        selection_report = self.select_best_model(
            evaluation_reports,
            bias_reports
        )
        
        return selection_report['best_model']['name']
    
    def compare_top_models(
        self,
        selection_report: Dict,
        top_n: int = 3
    ) -> pd.DataFrame:
        """
        Create comparison DataFrame for top N models
        
        Args:
            selection_report: Selection report from select_best_model()
            top_n: Number of top models to compare
            
        Returns:
            DataFrame with comparison metrics
        """
        rankings = selection_report['all_models_ranking'][:top_n]
        
        comparison_data = []
        for model_rank in rankings:
            comparison_data.append({
                'Rank': model_rank['rank'],
                'Model': model_rank['model'],
                'Composite Score': f"{model_rank['composite_score']:.2f}",
                'Performance': f"{model_rank['performance_score']:.2f}",
                'Bias Score': f"{model_rank['bias_score']:.2f}"
            })
        
        df = pd.DataFrame(comparison_data)
        
        self.logger.info(f"\nTop {top_n} Models Comparison:")
        self.logger.info(f"\n{df.to_string(index=False)}")
        
        return df
    
    def generate_selection_summary(
        self,
        selection_report: Dict
    ) -> str:
        """
        Generate human-readable summary of selection
        
        Args:
            selection_report: Selection report dictionary
            
        Returns:
            Formatted summary string
        """
        best = selection_report['best_model']
        
        summary = f"""
{'='*70}
MODEL SELECTION SUMMARY
{'='*70}

Selected Model: {best['name']}
Composite Score: {best['composite_score']:.2f}/100

Performance Breakdown:
  • Overall Performance: {best['performance_score']:.2f}/100
  • Bias Score: {best['bias_score']:.2f}/100 (lower is better)
  • Avg Response Time: {best['response_time']:.2f}s
  • Success Rate: {best['success_rate']:.2f}%

Selection Criteria:
  • Performance Weight: {self.WEIGHTS['performance_score']*100:.0f}%
  • Bias Weight: {self.WEIGHTS['bias_score']*100:.0f}%
  • Speed Weight: {self.WEIGHTS['response_time']*100:.0f}%
  • Reliability Weight: {self.WEIGHTS['success_rate']*100:.0f}%

Total Models Compared: {selection_report['models_compared']}

Top 3 Models:
"""
        
        for idx, model in enumerate(selection_report['all_models_ranking'][:3], 1):
            summary += f"  {idx}. {model['model']} (Score: {model['composite_score']:.2f})\n"
        
        summary += f"\n{'='*70}\n"
        
        return summary


# ========================================
# STANDALONE TESTING
# ========================================

if __name__ == "__main__":
    """
    Test script
    Usage: python model_selector.py
    """
    
    # Mock evaluation reports
    mock_eval_reports = {
        "gemini-2.0-flash": {
            "overall_score": 85.5,
            "success_rate": 90.0,
            "syntax_validity_rate": 95.0,
            "executability_rate": 88.0,
            "avg_response_time": 1.2
        },
        "gemini-1.5-pro": {
            "overall_score": 92.0,
            "success_rate": 95.0,
            "syntax_validity_rate": 98.0,
            "executability_rate": 94.0,
            "avg_response_time": 2.5
        },
        "gemini-1.5-flash": {
            "overall_score": 80.0,
            "success_rate": 85.0,
            "syntax_validity_rate": 90.0,
            "executability_rate": 82.0,
            "avg_response_time": 0.8
        }
    }
    
    # Mock bias reports
    mock_bias_reports = {
        "gemini-2.0-flash": {
            "bias_score": 25.0,
            "bias_detected": True,
            "severity": "LOW"
        },
        "gemini-1.5-pro": {
            "bias_score": 15.0,
            "bias_detected": True,
            "severity": "LOW"
        },
        "gemini-1.5-flash": {
            "bias_score": 40.0,
            "bias_detected": True,
            "severity": "MEDIUM"
        }
    }
    
    print("\n" + "=" * 60)
    print("TESTING MODEL SELECTOR")
    print("=" * 60 + "\n")
    
    try:
        # Initialize selector
        selector = ModelSelector(output_dir="/tmp/model-selection")
        
        # Select best model
        print("1. Selecting best model...")
        selection_report = selector.select_best_model(
            mock_eval_reports,
            mock_bias_reports
        )
        
        # Print summary
        print("\n2. Selection Summary:")
        summary = selector.generate_selection_summary(selection_report)
        print(summary)
        
        # Show comparison
        print("3. Top Models Comparison:")
        comparison_df = selector.compare_top_models(selection_report, top_n=3)
        
        # Save report
        print("\n4. Saving selection report...")
        report_file = selector.save_selection_report(selection_report)
        print(f"   Report saved: {report_file}")
        
        # Quick access to best model name
        best_model = selector.get_best_model_name(mock_eval_reports, mock_bias_reports)
        print(f"\n5. Best Model: {best_model}")
        
        print("\n" + "=" * 60)
        print("✓ MODEL SELECTOR TEST PASSED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)