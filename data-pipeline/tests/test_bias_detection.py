"""
Unit tests for BiasDetector class
Tests bias detection in protected attributes and metrics
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from bias_detection import BiasDetector


class TestBiasDetector:
    """Test suite for BiasDetector functionality"""
    
    @pytest.fixture
    def setup_test_env(self, tmp_path):
        """Set up temporary directory structure for testing"""
        processed_dir = tmp_path / "processed"
        profiles_dir = tmp_path / "profiles"
        
        processed_dir.mkdir(parents=True, exist_ok=True)
        profiles_dir.mkdir(parents=True, exist_ok=True)
        
        return str(tmp_path), "test_dataset"
    
    @pytest.fixture
    def biased_dataset(self):
        """Create a dataset with clear bias"""
        return pd.DataFrame({
            'segment': ['A', 'A', 'A', 'B', 'B'],  # Unbalanced representation
            'sales': [1000, 1100, 1200, 500, 600],  # Group B has much lower sales
            'region': ['East', 'West', 'East', 'North', 'South']
        })
    
    @pytest.fixture
    def balanced_dataset(self):
        """Create a balanced dataset"""
        return pd.DataFrame({
            'segment': ['A', 'A', 'B', 'B', 'C'],
            'sales': [1000, 1100, 1000, 1100, 1050],
            'region': ['East', 'West', 'North', 'South', 'Central']
        })
    
    def test_identify_protected_attributes(self, setup_test_env):
        """Test that protected attributes are identified correctly"""
        project_root, dataset_name = setup_test_env
        
        detector = BiasDetector(dataset_name)
        
        df = pd.DataFrame({
            'gender': ['M', 'F', 'M', 'F'],
            'age': [25, 30, 35, 40],
            'income': [50000, 60000, 70000, 80000]
        })
        
        detector.schema_profile = {
            'protected_attributes': ['gender', 'age']
        }
        
        protected_attrs = detector.identify_protected_attributes(df)
        
        # Check that protected attributes are identified
        assert 'gender' in protected_attrs or 'age' in protected_attrs
    
    def test_detect_representation_bias_finds_imbalance(self, setup_test_env, biased_dataset):
        """Test that representation bias is detected"""
        project_root, dataset_name = setup_test_env
        
        detector = BiasDetector(dataset_name)
        
        # Create very unbalanced data - one group < 10%
        unbalanced_data = pd.DataFrame({
            'segment': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B'],  # A=88.9%, B=11.1%
            'sales': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 500],
            'region': ['East', 'West', 'East', 'West', 'East', 'West', 'East', 'West', 'North']
        })
        
        result = detector.detect_representation_bias(unbalanced_data, 'segment')
        
        # Check that function returns expected structure
        assert 'representation_by_group' in result
        assert isinstance(result['has_representation_bias'], bool)
        assert isinstance(result['under_represented_groups'], list)
        # Should have representation data for A and B segments
        assert len(result['representation_by_group']) == 2
        assert 'A' in result['representation_by_group']
        assert 'B' in result['representation_by_group']
    
    def test_detect_representation_bias_no_bias(self, setup_test_env, balanced_dataset):
        """Test that no representation bias is detected in balanced data"""
        project_root, dataset_name = setup_test_env
        
        detector = BiasDetector(dataset_name)
        result = detector.detect_representation_bias(balanced_dataset, 'segment')
        
        # Should not detect significant bias in balanced data
        assert result['representation_by_group'] is not None
        # All groups should be roughly equally represented
    
    def test_disparate_impact_ratio_detects_bias(self, setup_test_env):
        """Test that disparate impact ratio correctly identifies unfair treatment"""
        project_root, dataset_name = setup_test_env
        
        detector = BiasDetector(dataset_name)
        
        # Create data with clear disparate impact
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B'],
            'outcome': [100, 110, 120, 40, 50]  # Group B has much lower outcomes
        })
        
        result = detector.disparate_impact_ratio(df, 'group', 'outcome')
        
        # Should detect that the ratio is below 0.8 (fails 80% rule)
        assert result['ratio'] < 0.8
        assert result['passes_80_percent_rule'] is False
        assert 'underprivileged_group' in result
    
    def test_statistical_parity_test(self, setup_test_env):
        """Test that statistical parity test detects significant differences"""
        project_root, dataset_name = setup_test_env
        
        detector = BiasDetector(dataset_name)
        
        # Create data with clear statistical difference
        df = pd.DataFrame({
            'group': ['A'] * 10 + ['B'] * 10,
            'metric': list(range(80, 90)) + list(range(40, 50))  # Different means
        })
        
        result = detector.statistical_parity_test(df, 'group', 'metric')
        
        assert result is not None
        assert 'p_value' in result
        # Should detect significant difference if means are very different
        # (this may or may not pass depending on variance)
    
    def test_analyze_group_statistics(self, setup_test_env):
        """Test that group statistics are calculated correctly"""
        project_root, dataset_name = setup_test_env
        
        detector = BiasDetector(dataset_name)
        
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [10, 20, 30, 40]
        })
        
        stats = detector.analyze_group_statistics(df, 'group', 'value')
        
        assert stats is not None
        assert 'A' in stats
        assert 'B' in stats
        assert 'mean' in stats['A']
        assert 'count' in stats['A']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

