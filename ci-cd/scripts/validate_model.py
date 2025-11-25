#!/usr/bin/env python3
"""
Validate Model Performance
Checks if model meets validation thresholds using best model responses
"""

import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Add paths
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "ci-cd" / "config" / "validation_thresholds.yaml"
outputs_dir = project_root / "outputs"

def load_thresholds() -> Dict:
    """Load validation thresholds from config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['validation_thresholds']

def find_latest_selection_report() -> Optional[Path]:
    """Find the latest model selection report in best-model-responses"""
    best_model_dir = outputs_dir / "best-model-responses"
    if not best_model_dir.exists():
        return None
    
    # Look for model_selection_report.json
    reports = list(best_model_dir.rglob("model_selection_report.json"))
    if reports:
        return max(reports, key=lambda p: p.stat().st_mtime)
    
    return None

def find_latest_accuracy_metrics() -> Optional[Path]:
    """Find the latest accuracy metrics"""
    best_model_dir = outputs_dir / "best-model-responses"
    if not best_model_dir.exists():
        return None
    
    # Look for accuracy_metrics.json
    accuracy_files = list(best_model_dir.rglob("accuracy_metrics.json"))
    if accuracy_files:
        return max(accuracy_files, key=lambda p: p.stat().st_mtime)
    
    return None

def validate_performance(metrics: Dict, thresholds: Dict) -> List[str]:
    """Validate performance metrics from selection report"""
    errors = []
    perf_thresholds = thresholds['performance']
    
    # Get metrics from best_model section
    best_model = metrics.get('best_model', {})
    
    # Use composite_score as overall_score
    overall_score = best_model.get('composite_score', 0)
    if overall_score < perf_thresholds['min_overall_score']:
        errors.append(
            f"Composite score {overall_score:.2f} below threshold "
            f"{perf_thresholds['min_overall_score']}"
        )
    
    # Use performance_score
    performance_score = best_model.get('performance_score', 0)
    if performance_score < perf_thresholds.get('min_performance_score', 0):
        errors.append(
            f"Performance score {performance_score:.2f} below threshold "
            f"{perf_thresholds.get('min_performance_score', 0)}"
        )
    
    success_rate = best_model.get('success_rate', 0)
    if success_rate < perf_thresholds['min_success_rate']:
        errors.append(
            f"Success rate {success_rate:.2f}% below threshold "
            f"{perf_thresholds['min_success_rate']}%"
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
        
        # Load selection report
        selection_report_path = find_latest_selection_report()
        if not selection_report_path:
            print("✗ No model selection report found")
            return 1
        
        print(f"✓ Found selection report: {selection_report_path.name}")
        
        with open(selection_report_path, 'r') as f:
            selection_data = json.load(f)
        
        best_model_name = selection_data.get('best_model', {}).get('name', 'unknown')
        print(f"✓ Best model: {best_model_name}")
        
        # Validate performance
        print("\nValidating performance metrics...")
        perf_errors = validate_performance(selection_data, thresholds)
        
        # Validate execution
        print("Validating execution metrics...")
        accuracy_file = find_latest_accuracy_metrics()
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
        best_model = selection_data.get('best_model', {})
        validation_report = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "status": "passed" if not all_errors else "failed",
            "errors": all_errors,
            "metrics": {
                "composite_score": best_model.get('composite_score', 0),
                "performance_score": best_model.get('performance_score', 0),
                "success_rate": best_model.get('success_rate', 0),
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
            print(f"  ✓ Composite Score: {best_model.get('composite_score', 0):.2f}")
            print(f"  ✓ Performance Score: {best_model.get('performance_score', 0):.2f}")
            print(f"  ✓ Success Rate: {best_model.get('success_rate', 0):.2f}%")
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
