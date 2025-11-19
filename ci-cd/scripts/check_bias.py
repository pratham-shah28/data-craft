#!/usr/bin/env python3
"""
Check Bias Detection Results
Validates that bias is within acceptable thresholds
"""

import sys
import json
import yaml
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "ci-cd" / "config" / "validation_thresholds.yaml"
outputs_dir = project_root / "outputs"

def load_thresholds():
    """Load bias thresholds"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['validation_thresholds']['bias']

def find_latest_bias_comparison() -> Path:
    """Find the latest bias comparison report"""
    bias_dir = outputs_dir / "bias"
    if not bias_dir.exists():
        raise FileNotFoundError(f"Bias directory not found: {bias_dir}")
    
    reports = list(bias_dir.glob("bias_comparison_*.json"))
    if not reports:
        raise FileNotFoundError("No bias comparison reports found")
    
    return max(reports, key=lambda p: p.stat().st_mtime)

def main():
    """Check bias detection results"""
    print("=" * 70)
    print("BIAS DETECTION CHECK")
    print("=" * 70)
    
    try:
        thresholds = load_thresholds()
        print("✓ Loaded bias thresholds")
        
        bias_report_path = find_latest_bias_comparison()
        print(f"✓ Found bias report: {bias_report_path.name}")
        
        with open(bias_report_path, 'r') as f:
            bias_data = json.load(f)
        
        # Check best model (least biased)
        if 'bias_comparison' in bias_data and bias_data['bias_comparison']:
            best_model = bias_data['bias_comparison'][0]  # Sorted by bias score (ascending)
            bias_score = best_model.get('bias_score', 100)
            severity = best_model.get('severity', 'HIGH')
            
            print(f"\nBest Model: {best_model.get('model')}")
            print(f"  Bias Score: {bias_score:.2f}/100")
            print(f"  Severity: {severity}")
            
            errors = []
            
            # Check bias score threshold
            if bias_score > thresholds['max_bias_score']:
                errors.append(
                    f"Bias score {bias_score:.2f} exceeds threshold "
                    f"{thresholds['max_bias_score']}"
                )
            
            # Check severity threshold
            severity_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
            max_severity_level = severity_levels.get(thresholds['max_severity'], 2)
            model_severity_level = severity_levels.get(severity, 3)
            
            if model_severity_level > max_severity_level:
                errors.append(
                    f"Bias severity '{severity}' exceeds threshold "
                    f"'{thresholds['max_severity']}'"
                )
            
            # Save check report
            check_report = {
                "status": "passed" if not errors else "failed",
                "bias_score": bias_score,
                "severity": severity,
                "errors": errors,
                "model": best_model.get('model')
            }
            
            check_dir = outputs_dir / "validation"
            check_dir.mkdir(parents=True, exist_ok=True)
            check_file = check_dir / "bias_check_report.json"
            
            with open(check_file, 'w') as f:
                json.dump(check_report, f, indent=2)
            
            print("\n" + "=" * 70)
            if errors:
                print("BIAS CHECK FAILED")
                print("=" * 70)
                for error in errors:
                    print(f"  ✗ {error}")
                print(f"\nReport saved: {check_file}")
                return 1
            else:
                print("BIAS CHECK PASSED")
                print("=" * 70)
                print(f"  ✓ Bias score within acceptable range")
                print(f"  ✓ Severity level acceptable")
                print(f"\nReport saved: {check_file}")
                return 0
        else:
            print("⚠ No bias comparison data found")
            return 0
            
    except Exception as e:
        print(f"\n✗ BIAS CHECK ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

