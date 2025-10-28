"""
Pytest configuration and shared fixtures
Sets up test environment and common fixtures
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_config():
    """Create a test configuration"""
    return {
        'data': {
            'raw_path': 'data/raw/',
            'processed_path': 'data/processed/',
            'validated_path': 'data/validated/',
            'supported_formats': ['csv', 'xlsx', 'json']
        },
        'gcp': {
            'bucket_name': 'test-bucket',
            'project_id': 'test-project'
        },
        'validation': {
            'null_threshold': 0.15,
            'duplicate_threshold': 0.05,
            'min_rows': 10,
            'min_columns': 2
        },
        'bias': {
            'auto_detect_protected_attrs': True,
            'sensitive_keywords': ['gender', 'age', 'race', 'segment', 'region']
        },
        'cleaning': {
            'auto_detect_types': True,
            'drop_duplicates': True,
            'handle_outliers': True,
            'outlier_method': 'iqr',
            'outlier_threshold': 3
        }
    }

