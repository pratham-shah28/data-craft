#!/usr/bin/env python
"""
Simple test runner for MLOps data pipeline
Usage: python run_tests.py [--all | --validation | --cleaning | --bias | --schema | --utils]
"""

import sys
import subprocess
from pathlib import Path

def run_tests(test_type='all'):
    """Run specific tests or all tests"""
    
    test_files = {
        'all': 'tests/',
        'validation': 'tests/test_data_validation.py',
        'cleaning': 'tests/test_data_cleaning.py',
        'bias': 'tests/test_bias_detection.py',
        'schema': 'tests/test_schema_detector.py',
        'utils': 'tests/test_utils.py'
    }
    
    if test_type not in test_files:
        print(f"Unknown test type: {test_type}")
        print(f"Available: {', '.join(test_files.keys())}")
        return 1
    
    test_path = test_files[test_type]
    
    print(f"\n{'='*60}")
    print(f"Running tests: {test_path}")
    print(f"{'='*60}\n")
    
    # Run pytest with verbose output
    cmd = ['python', '-m', 'pytest', test_path, '-v', '--tb=short']
    
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_type = sys.argv[1].replace('--', '')
    else:
        test_type = 'all'
    
    exit_code = run_tests(test_type)
    sys.exit(exit_code)

