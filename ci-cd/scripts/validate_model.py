#!/usr/bin/env python3
"""
Validate Model Performance
Checks if model meets validation thresholds
"""

import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List

# Add paths
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "ci-cd" / "config" / "validation_thresholds.yaml"
outputs_dir = project_root / "outputs"

def load_thresholds() -> Dict:
    """Load validation thresholds from config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['validation_thresholds']

def find_latest_evaluation_report() -> Path:
    """Find the latest evaluation report"""
    eval_dir = outputs_dir / "evaluation"
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")
    
    # Find comparison report (most recent)
    comparison_reports = list(eval_dir.glob("model_comparison_*.json"))
    if comparison_reports:
        return max(comparison_reports, key=lambda p: p.stat().st_mtime)
    
    # Fallback: find any evaluation report
    reports = list(eval_dir.glob("*_evaluation_*.json"))
    if reports:
        return max(reports, key=lambda p: p.stat().st_mtime)
    
    raise FileNotFoundError("No evaluation reports found")

def find_latest_selection_report() -> Path:
    """Find the latest model selection report"""
    selection_dir = outputs_dir / "model-selection"
    if not selection_dir.exists():
        raise FileNotFoundError(f"Selection directory not found: {selection_dir}")
    
    reports = list(selection_dir.glob("model_selection_*.json"))
    if not reports:
        raise FileNotFoundError("No model selection reports found")
    
    return max(reports, key=lambda p: p.stat().st_mtime)

def find_latest_execution_results() -> Path:
    """Find the latest execution results"""
    best_model_dir = outputs_dir / "best-model-responses"
    if not best_model_dir.exists():
        return None
    
    # Look for accuracy_metrics.json
    accuracy_files = list(best_model_dir.rglob("accuracy_metrics.json"))
    if accuracy_files:
        return max(accuracy_files, key=lambda p: p.stat().st_mtime)
    
    return None

def validate_performance(metrics: Dict, thresholds: Dict) -> List[str]:
    """Validate performance metrics"""
    errors = []
    perf_thresholds = thresholds['performance']
    
    overall_score = metrics.get('overall_score', 0)
    if overall_score < perf_thresholds['min_overall_score']:
        errors.append(
            f"Overall score {overall_score:.2f} below threshold "
            f"{perf_thresholds['min_overall_score']}"
        )
    
    success_rate = metrics.get('success_rate', 0)
    if success_rate < perf_thresholds['min_success_rate']:
        errors.append(
            f"Success rate {success_rate:.2f}% below threshold "
            f"{perf_thresholds['min_success_rate']}%"
        )
    
    syntax_validity = metrics.get('syntax_validity_rate', 0)
    if syntax_validity < perf_thresholds['min_syntax_validity']:
        errors.append(
            f"Syntax validity {syntax_validity:.2f}% below threshold "
            f"{perf_thresholds['min_syntax_validity']}%"
        )
    
    intent_matching = metrics.get('intent_matching_rate', 0)
    if intent_matching < perf_thresholds['min_intent_matching']:
        errors.append(
            f"Intent matching {intent_matching:.2f}% below threshold "
            f"{perf_thresholds['min_intent_matching']}%"
        )
    
    return errors

def validate_execution(accuracy_metrics: Dict, thresholds: Dict) -> List[str]:
    """Validate execution metrics"""
    errors = []
    exec_thresholds = thresholds['execution']
    
    if not accuracy_metrics:
        return ["No execution results found"]
    
    exec_success = accuracy_metrics.get('execution_success_rate', 0)
    if exec_success < exec_thresholds['min_execution_success_rate']:
        errors.append(
            f"Execution success rate {exec_success:.2f}% below threshold "
            f"{exec_thresholds['min_execution_success_rate']}%"
        )
    
    validity_rate = accuracy_metrics.get('result_validity_rate', 0)
    if validity_rate < exec_thresholds['min_result_validity_rate']:
        errors.append(
            f"Result validity rate {validity_rate:.2f}% below threshold "
            f"{exec_thresholds['min_result_validity_rate']}%"
        )
    
    overall_accuracy = accuracy_metrics.get('overall_accuracy', 0)
    if overall_accuracy < exec_thresholds['min_overall_accuracy']:
        errors.append(
            f"Overall accuracy {overall_accuracy:.2f}% below threshold "
            f"{exec_thresholds['min_overall_accuracy']}%"
        )
    
    return errors

def main():
    """Main validation function"""
    print("=" * 70)
    print("MODEL VALIDATION")
    print("=" * 70)
    
    try:
        # Load thresholds
        thresholds = load_thresholds()
        print("✓ Loaded validation thresholds")
        
        # Load evaluation report
        eval_report_path = find_latest_evaluation_report()
        print(f"✓ Found evaluation report: {eval_report_path.name}")
        
        with open(eval_report_path, 'r') as f:
            eval_data = json.load(f)
        
        # Get best model metrics from comparison report
        if 'detailed_comparison' in eval_data:
            best_model_metrics = eval_data['detailed_comparison'][0]
            print(f"✓ Best model: {best_model_metrics.get('model')}")
        else:
            # Fallback: use first model in report
            best_model_metrics = eval_data
        
        # Validate performance
        print("\nValidating performance metrics...")
        perf_errors = validate_performance(best_model_metrics, thresholds)
        
        # Validate execution
        print("Validating execution metrics...")
        accuracy_file = find_latest_execution_results()
        accuracy_metrics = None
        if accuracy_file:
            with open(accuracy_file, 'r') as f:
                accuracy_metrics = json.load(f)
            print(f"✓ Found execution results: {accuracy_file.name}")
        else:
            print("⚠ No execution results found")
        
        exec_errors = validate_execution(accuracy_metrics, thresholds)
        
        # Combine errors
        all_errors = perf_errors + exec_errors
        
        # Save validation report
        validation_report = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "status": "passed" if not all_errors else "failed",
            "errors": all_errors,
            "metrics": {
                "overall_score": best_model_metrics.get('overall_score', 0),
                "success_rate": best_model_metrics.get('success_rate', 0),
                "syntax_validity": best_model_metrics.get('syntax_validity', 0),
                "intent_matching": best_model_metrics.get('intent_matching', 0),
                "execution_accuracy": accuracy_metrics.get('overall_accuracy', 0) if accuracy_metrics else None
            }
        }
        
        validation_dir = outputs_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        validation_file = validation_dir / "validation_report.json"
        
        with open(validation_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        # Print results
        print("\n" + "=" * 70)
        if all_errors:
            print("VALIDATION FAILED")
            print("=" * 70)
            for error in all_errors:
                print(f"  ✗ {error}")
            print(f"\nValidation report saved: {validation_file}")
            return 1
        else:
            print("VALIDATION PASSED")
            print("=" * 70)
            print(f"  ✓ Overall Score: {best_model_metrics.get('overall_score', 0):.2f}")
            print(f"  ✓ Success Rate: {best_model_metrics.get('success_rate', 0):.2f}%")
            if accuracy_metrics:
                print(f"  ✓ Execution Accuracy: {accuracy_metrics.get('overall_accuracy', 0):.2f}%")
            print(f"\nValidation report saved: {validation_file}")
            return 0
            
    except Exception as e:
        print(f"\n✗ VALIDATION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

