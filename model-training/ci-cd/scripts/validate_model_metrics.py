#!/usr/bin/env python3
"""
Model Metrics Validation Script
Validates model metrics against predefined thresholds
Returns exit code 0 if validation passes, 1 if fails

Usage:
    python validate_model_metrics.py [metrics_file_path]
"""

import json
import sys
from pathlib import Path
import yaml

def load_config():
    """Load CI/CD configuration"""
    config_path = Path(__file__).parent.parent / "config" / "ci_cd_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_model(metrics_file: str, config: dict) -> tuple[bool, dict]:
    """
    Validate model metrics against thresholds
    
    Args:
        metrics_file: Path to metrics JSON file
        config: CI/CD configuration dictionary
        
    Returns:
        Tuple of (all_passed: bool, check_results: dict)
    """
    if not Path(metrics_file).exists():
        print(f"ERROR: Metrics file not found: {metrics_file}")
        return False, {}
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    thresholds = config['validation']['thresholds']
    
    # Extract metrics
    best_model = metrics.get('best_model', {})
    accuracy_metrics = metrics.get('accuracy', {})
    
    # Perform validation checks
    checks = {
        'composite_score': best_model.get('composite_score', 0) >= thresholds['min_composite_score'],
        'performance_score': best_model.get('performance_score', 0) >= thresholds['min_performance_score'],
        'bias_score': best_model.get('bias_score', 100) <= thresholds['max_bias_score'],
        'success_rate': best_model.get('success_rate', 0) >= thresholds['min_success_rate'],
        'overall_accuracy': accuracy_metrics.get('overall_accuracy', 0) >= thresholds['min_overall_accuracy']
    }
    
    # Get actual values for reporting
    check_details = {
        'composite_score': {
            'passed': checks['composite_score'],
            'value': best_model.get('composite_score', 0),
            'threshold': thresholds['min_composite_score']
        },
        'performance_score': {
            'passed': checks['performance_score'],
            'value': best_model.get('performance_score', 0),
            'threshold': thresholds['min_performance_score']
        },
        'bias_score': {
            'passed': checks['bias_score'],
            'value': best_model.get('bias_score', 100),
            'threshold': thresholds['max_bias_score'],
            'note': 'Lower is better'
        },
        'success_rate': {
            'passed': checks['success_rate'],
            'value': best_model.get('success_rate', 0),
            'threshold': thresholds['min_success_rate']
        },
        'overall_accuracy': {
            'passed': checks['overall_accuracy'],
            'value': accuracy_metrics.get('overall_accuracy', 0),
            'threshold': thresholds['min_overall_accuracy']
        }
    }
    
    all_passed = all(checks.values())
    
    return all_passed, check_details

def print_validation_results(check_details: dict):
    """Print validation results in a readable format"""
    print("\n" + "=" * 70)
    print("MODEL VALIDATION RESULTS")
    print("=" * 70)
    
    for check_name, details in check_details.items():
        status = "✓ PASS" if details['passed'] else "✗ FAIL"
        value = details['value']
        threshold = details['threshold']
        
        if check_name == 'bias_score':
            comparison = "≤" if details['passed'] else ">"
            print(f"{status} {check_name.replace('_', ' ').title()}: {value:.2f} {comparison} {threshold:.2f} (lower is better)")
        else:
            comparison = "≥" if details['passed'] else "<"
            print(f"{status} {check_name.replace('_', ' ').title()}: {value:.2f} {comparison} {threshold:.2f}")
    
    print("=" * 70)
    
    all_passed = all(d['passed'] for d in check_details.values())
    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
    else:
        print("✗ VALIDATION FAILED - One or more checks did not meet thresholds")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Get metrics file path
    if len(sys.argv) > 1:
        metrics_file = sys.argv[1]
    else:
        # Default path
        metrics_file = config['paths']['model_training_output'] + "/pipeline_summary.json"
    
    # Validate model
    all_passed, check_details = validate_model(metrics_file, config)
    
    # Print results
    print_validation_results(check_details)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

