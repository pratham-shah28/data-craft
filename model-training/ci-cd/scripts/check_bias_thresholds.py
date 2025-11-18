#!/usr/bin/env python3
"""
Bias Threshold Checker Script
Checks if bias scores exceed acceptable thresholds
Blocks deployment if bias is too high

Usage:
    python check_bias_thresholds.py [bias_report_file]
"""

import json
import sys
from pathlib import Path
import glob
import yaml

def load_config():
    """Load CI/CD configuration"""
    config_path = Path(__file__).parent.parent / "config" / "ci_cd_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_latest_bias_report(bias_output_dir: str) -> str:
    """Find the latest bias comparison report"""
    pattern = f"{bias_output_dir}/bias_comparison_*.json"
    files = glob.glob(pattern)
    
    if not files:
        # Try individual model bias reports
        pattern = f"{bias_output_dir}/*_bias_*.json"
        files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Return most recently modified
    return max(files, key=lambda p: Path(p).stat().st_mtime)

def check_bias(bias_report_file: str, config: dict) -> tuple[bool, dict]:
    """
    Check bias against thresholds
    
    Args:
        bias_report_file: Path to bias report JSON file
        config: CI/CD configuration dictionary
        
    Returns:
        Tuple of (all_passed: bool, check_results: dict)
    """
    if not bias_report_file or not Path(bias_report_file).exists():
        print(f"ERROR: Bias report file not found: {bias_report_file}")
        return False, {}
    
    with open(bias_report_file, 'r') as f:
        bias_data = json.load(f)
    
    thresholds = config['bias']['thresholds']
    
    # Handle both comparison report and individual model report formats
    if 'bias_comparison' in bias_data:
        # Comparison report - check all models
        bias_scores = [m['bias_score'] for m in bias_data['bias_comparison']]
        severities = [m['severity'] for m in bias_data['bias_comparison']]
        num_biases_list = [m['num_biases'] for m in bias_data['bias_comparison']]
        
        # Use worst case across all models
        max_bias_score = max(bias_scores) if bias_scores else 100
        max_severity = max(severities, key=lambda s: ['LOW', 'MEDIUM', 'HIGH'].index(s) if s in ['LOW', 'MEDIUM', 'HIGH'] else 0)
        max_num_biases = max(num_biases_list) if num_biases_list else 0
    else:
        # Individual model report
        max_bias_score = bias_data.get('bias_score', 100)
        max_severity = bias_data.get('severity', 'UNKNOWN')
        max_num_biases = len(bias_data.get('biases', []))
    
    # Perform checks
    checks = {
        'bias_score': max_bias_score <= thresholds['max_bias_score'],
        'severity': max_severity != 'HIGH',
        'num_biases': max_num_biases <= thresholds['max_individual_biases']
    }
    
    # Get check details
    severity_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'UNKNOWN': 0}
    max_severity_level = severity_levels.get(max_severity, 0)
    threshold_severity_level = severity_levels.get(thresholds['max_severity'], 0)
    
    check_details = {
        'bias_score': {
            'passed': checks['bias_score'],
            'value': max_bias_score,
            'threshold': thresholds['max_bias_score'],
            'note': 'Lower is better'
        },
        'severity': {
            'passed': checks['severity'],
            'value': max_severity,
            'threshold': thresholds['max_severity'],
            'note': 'HIGH severity is not allowed'
        },
        'num_biases': {
            'passed': checks['num_biases'],
            'value': max_num_biases,
            'threshold': thresholds['max_individual_biases']
        }
    }
    
    all_passed = all(checks.values())
    
    return all_passed, check_details

def print_bias_results(check_details: dict):
    """Print bias check results in a readable format"""
    print("\n" + "=" * 70)
    print("BIAS THRESHOLD CHECK RESULTS")
    print("=" * 70)
    
    for check_name, details in check_details.items():
        status = "✓ PASS" if details['passed'] else "✗ FAIL"
        value = details['value']
        threshold = details['threshold']
        
        if check_name == 'bias_score':
            comparison = "≤" if details['passed'] else ">"
            print(f"{status} {check_name.replace('_', ' ').title()}: {value:.2f} {comparison} {threshold:.2f} (lower is better)")
        elif check_name == 'severity':
            print(f"{status} {check_name.replace('_', ' ').title()}: {value} (max allowed: {threshold})")
        else:
            comparison = "≤" if details['passed'] else ">"
            print(f"{status} {check_name.replace('_', ' ').title()}: {value} {comparison} {threshold}")
    
    print("=" * 70)
    
    all_passed = all(d['passed'] for d in check_details.values())
    if all_passed:
        print("✓ ALL BIAS CHECKS PASSED")
    else:
        print("✗ BIAS CHECK FAILED - Bias thresholds exceeded! Deployment blocked.")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Get bias report file path
    if len(sys.argv) > 1:
        bias_file = sys.argv[1]
    else:
        # Find latest bias report
        bias_output_dir = config['paths']['bias_output']
        bias_file = find_latest_bias_report(bias_output_dir)
        
        if not bias_file:
            print(f"ERROR: No bias report found in {bias_output_dir}")
            sys.exit(1)
    
    # Check bias
    all_passed, check_details = check_bias(bias_file, config)
    
    # Print results
    print_bias_results(check_details)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)

