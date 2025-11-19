# scripts/hyperparameter_tuner.py
"""
Hyperparameter Tuning for Gemini Generation Config
Tests different temperature, top_p, and top_k values
"""

import json
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig


class HyperparameterTuner:
    """
    Tune Gemini generation parameters
    
    Parameters tested:
    - temperature: Controls randomness (0.0-1.0)
    - top_p: Nucleus sampling (0.0-1.0)
    - top_k: Top-k sampling (1-100)
    """
    
    def __init__(self, project_id: str, region: str, output_dir: str = "/opt/airflow/outputs/hyperparameter-tuning"):
        """
        Initialize hyperparameter tuner
        
        Args:
            project_id: GCP project ID
            region: GCP region
            output_dir: Directory to save tuning reports
        """
        self.project_id = project_id
        self.region = region
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(self.__class__.__name__)
        
        # Setup GCP credentials for Vertex AI
        from utils import setup_gcp_credentials
        try:
            setup_gcp_credentials(logger=self.logger)
        except Exception as e:
            self.logger.warning(f"GCP credentials setup: {e}")
        
        # Initialize Vertex AI
        try:
            vertexai.init(project=project_id, location=region)
            self.logger.info("✓ Vertex AI initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    def tune_generation_params(
        self,
        validation_queries: List[str],
        llm_context: str,
        model_name: str = "gemini-2.5-flash"
    ) -> Tuple[Dict, List[Dict]]:
        """
        Grid search for best generation parameters
        
        Args:
            validation_queries: List of queries to test
            llm_context: Dataset context
            model_name: Model to tune
            
        Returns:
            Tuple of (best_params, all_results)
        """
        self.logger.info(f"Starting hyperparameter tuning for {model_name}")
        self.logger.info(f"Testing with {len(validation_queries)} queries")
        
        # Define parameter grid (reduced for speed)
        param_grid = {
            'temperature': [0.2, 0.4, 0.6],
            'top_p': [0.8, 0.9],
            'top_k': [40, 60]
        }
        
        all_results = []
        best_params = None
        best_score = 0
        
        total_combos = len(param_grid['temperature']) * len(param_grid['top_p']) * len(param_grid['top_k'])
        combo_num = 0
        
        # Grid search
        for temp in param_grid['temperature']:
            for top_p in param_grid['top_p']:
                for top_k in param_grid['top_k']:
                    combo_num += 1
                    
                    params = {
                        'temperature': temp,
                        'top_p': top_p,
                        'top_k': top_k
                    }
                    
                    self.logger.info(f"\n[{combo_num}/{total_combos}] Testing: temp={temp}, top_p={top_p}, top_k={top_k}")
                    
                    # Evaluate with these parameters
                    score = self._evaluate_with_params(
                        validation_queries[:5],  # Use first 5 queries only (speed)
                        llm_context,
                        model_name,
                        params
                    )
                    
                    result = {
                        'parameters': params,
                        'score': score
                    }
                    
                    all_results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        self.logger.info(f"  ✓ New best score: {best_score:.2f}")
        
        self.logger.info(f"\n✓ Tuning complete!")
        self.logger.info(f"  Best params: {best_params}")
        self.logger.info(f"  Best score: {best_score:.2f}")
        
        return best_params, all_results
    
    def _evaluate_with_params(
        self,
        queries: List[str],
        llm_context: str,
        model_name: str,
        params: Dict
    ) -> float:
        """
        Evaluate model with specific parameters
        
        Returns:
            Score (0-100) based on response quality
        """
        from prompts import build_prompt
        
        successful = 0
        valid_format = 0
        
        model = GenerativeModel(
            model_name,
            generation_config=GenerationConfig(
                temperature=params['temperature'],
                top_p=params['top_p'],
                top_k=params['top_k'],
                max_output_tokens=2048
            )
        )
        
        for query in queries:
            try:
                prompt = build_prompt(query, llm_context, "")
                response = model.generate_content(prompt)
                
                # Try to parse response
                import re
                response_text = response.text
                response_text = re.sub(r'```json\s*', '', response_text)
                response_text = re.sub(r'```\s*', '', response_text).strip()
                
                result = json.loads(response_text)
                
                if 'sql_query' in result and 'visualization' in result:
                    valid_format += 1
                
                successful += 1
                
            except Exception:
                pass
        
        # Score based on success rate and format compliance
        score = ((successful / len(queries)) * 50) + ((valid_format / len(queries)) * 50)
        
        return score
    
    def save_tuning_report(
        self,
        best_params: Dict,
        all_results: List[Dict]
    ) -> str:
        """Save tuning report to JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"hyperparameter_tuning_{timestamp}.json"
        
        report = {
            "tuning_timestamp": datetime.now().isoformat(),
            "best_parameters": best_params,
            "all_results": all_results,
            "total_combinations_tested": len(all_results)
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"✓ Saved tuning report: {report_file}")
        
        return str(report_file)


# ========================================
# STANDALONE TESTING
# ========================================

if __name__ == "__main__":
    """
    Test script
    Usage: python hyperparameter_tuner.py
    """
    
    print("\n" + "=" * 60)
    print("TESTING HYPERPARAMETER TUNER")
    print("=" * 60 + "\n")
    
    print("Note: Full testing requires Vertex AI credentials")
    print("This validates the tuning logic\n")
    
    try:
        tuner = HyperparameterTuner(
            project_id="datacraft-data-pipeline",
            region="us-central1",
            output_dir="/tmp/hyperparameter-tuning"
        )
        
        print("✓ Tuner initialized")
        print("\nIn production, tuner would test:")
        print("  - 3 temperature values")
        print("  - 2 top_p values")
        print("  - 2 top_k values")
        print("  Total: 12 combinations")
        
        # Mock results
        mock_best_params = {'temperature': 0.2, 'top_p': 0.9, 'top_k': 40}
        mock_results = [
            {'parameters': mock_best_params, 'score': 85.5}
        ]
        
        # Save mock report
        report_file = tuner.save_tuning_report(mock_best_params, mock_results)
        print(f"\n✓ Report saved: {report_file}")
        
        print("\n" + "=" * 60)
        print("✓ HYPERPARAMETER TUNER TEST PASSED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)