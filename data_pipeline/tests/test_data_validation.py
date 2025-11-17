"""
Unit tests for DataValidator class
Tests basic structure validation, null checks, duplicates, and anomaly detection
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from data_validation import DataValidator


class TestDataValidator:
    """Test suite for DataValidator functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, None, 35, 40],
            'score': [85.5, 90.0, 88.2, 92.0, 87.5],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
    
    @pytest.fixture
    def setup_test_env(self, tmp_path, sample_data):
        """Set up temporary directory structure for testing"""
        # Create temporary directory structure
        raw_dir = tmp_path / "data" / "raw"
        processed_dir = tmp_path / "processed"
        profiles_dir = tmp_path / "profiles"
        
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sample data
        dataset_name = "test_orders"
        sample_data.to_csv(raw_dir / f"{dataset_name}.csv", index=False)
        
        # Create a minimal schema profile
        import json
        schema_profile = {
            "dataset_name": dataset_name,
            "column_types": {
                "identifier": ["id"],
                "categorical": ["category"],
                "continuous_numeric": ["score"],
                "discrete_numeric": ["age"]
            },
            "protected_attributes": ["category"]
        }
        
        with open(profiles_dir / f"{dataset_name}_profile.json", 'w') as f:
            json.dump(schema_profile, f)
        
        return str(tmp_path), dataset_name
    
    def test_validate_basic_structure_passes(self, setup_test_env):
        """Test that validation passes for data with sufficient rows/columns"""
        project_root, dataset_name = setup_test_env
        
        validator = DataValidator(dataset_name)
        
        # Modify config to allow small datasets for testing
        validator.config['validation']['min_rows'] = 5
        validator.config['validation']['min_columns'] = 5
        
        # Create test data with enough rows and columns
        test_data = pd.DataFrame({
            'col1': range(10),
            'col2': range(10),
            'col3': range(10),
            'col4': range(10),
            'col5': range(10)
        })
        
        result = validator.validate_basic_structure(test_data)
        assert result is True
        assert validator.validation_report['checks']['structure']['valid'] is True
    
    def test_validate_basic_structure_fails_small_dataset(self, setup_test_env):
        """Test that validation fails for datasets that are too small"""
        project_root, dataset_name = setup_test_env
        
        validator = DataValidator(dataset_name)
        validator.config['validation']['min_rows'] = 10
        validator.config['validation']['min_columns'] = 5
        
        test_data = pd.DataFrame({'col1': range(5), 'col2': range(5)})
        
        result = validator.validate_basic_structure(test_data)
        assert result is False
        assert validator.validation_report['checks']['structure']['valid'] is False
    
    def test_check_nulls_passes(self, setup_test_env):
        """Test null check passes when null percentage is below threshold"""
        project_root, dataset_name = setup_test_env
        
        validator = DataValidator(dataset_name)
        validator.config['validation']['null_threshold'] = 0.5  # Allow up to 50% nulls
        
        test_data = pd.DataFrame({
            'col1': [1, 2, None, 4, 5],
            'col2': [10, 20, 30, 40, 50]
        })
        
        result = validator.check_nulls(test_data)
        assert result is True
        assert validator.validation_report['checks']['nulls']['valid'] is True
    
    def test_check_nulls_fails_high_nulls(self, setup_test_env):
        """Test null check fails when null percentage exceeds threshold"""
        project_root, dataset_name = setup_test_env
        
        validator = DataValidator(dataset_name)
        validator.config['validation']['null_threshold'] = 0.2  # Max 20% nulls
        
        test_data = pd.DataFrame({
            'col1': [1, None, None, None, None],  # 80% nulls
            'col2': [10, 20, 30, 40, 50]
        })
        
        result = validator.check_nulls(test_data)
        assert result is False
        assert validator.validation_report['checks']['nulls']['valid'] is False
        assert len(validator.validation_report['checks']['nulls']['columns_exceeding_threshold']) > 0
    
    def test_check_duplicates_passes(self, setup_test_env):
        """Test duplicate check passes when duplicate percentage is below threshold"""
        project_root, dataset_name = setup_test_env
        
        validator = DataValidator(dataset_name)
        validator.config['validation']['duplicate_threshold'] = 0.3  # Allow up to 30% duplicates
        
        test_data = pd.DataFrame({
            'col1': [1, 2, 3, 3, 4],
            'col2': ['a', 'b', 'c', 'c', 'd']
        })
        
        result = validator.check_duplicates(test_data)
        assert result
        assert validator.validation_report['checks']['duplicates']['valid'] is True
    
    def test_check_duplicates_fails_high_duplicates(self, setup_test_env):
        """Test duplicate check fails when duplicate percentage exceeds threshold"""
        project_root, dataset_name = setup_test_env
        
        validator = DataValidator(dataset_name)
        validator.config['validation']['duplicate_threshold'] = 0.1  # Max 10% duplicates
        
        test_data = pd.DataFrame({
            'col1': [1, 2, 3, 3, 3],
            'col2': ['a', 'b', 'c', 'c', 'c']
        })
        
        result = validator.check_duplicates(test_data)
        assert not result
        assert validator.validation_report['checks']['duplicates']['valid'] is False
    
    def test_detect_anomalies_finds_outliers(self, setup_test_env):
        """Test that anomaly detection identifies outliers correctly"""
        project_root, dataset_name = setup_test_env
        
        validator = DataValidator(dataset_name)
        validator.schema_profile = None  # Use fallback detection
        
        # Create data with obvious outliers
        test_data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000]  # 1000 is an outlier
        })
        
        anomalies = validator.detect_anomalies(test_data)
        
        assert anomalies is not None
        assert 'value' in anomalies
        assert anomalies['value']['outlier_count'] > 0
    
    def test_detect_anomalies_no_outliers(self, setup_test_env):
        """Test that anomaly detection doesn't find outliers in normal data"""
        project_root, dataset_name = setup_test_env
        
        validator = DataValidator(dataset_name)
        validator.schema_profile = None
        
        # Create normal data without outliers
        test_data = pd.DataFrame({
            'value': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        })
        
        anomalies = validator.detect_anomalies(test_data)
        
        # Should detect anomalies structure but with no outliers
        assert anomalies is not None
        assert len(anomalies.get('columns_with_anomalies', [])) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

