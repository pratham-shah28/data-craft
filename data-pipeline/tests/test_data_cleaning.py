"""
Unit tests for DataCleaner class
Tests cleaning operations: normalization, type standardization, missing values, outliers, duplicates
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
from data_cleaning import DataCleaner


class TestDataCleaner:
    """Test suite for DataCleaner functionality"""
    
    @pytest.fixture
    def setup_test_env(self, tmp_path):
        """Set up temporary directory structure for testing"""
        processed_dir = tmp_path / "processed"
        validated_dir = tmp_path / "validated"
        profiles_dir = tmp_path / "profiles"
        
        processed_dir.mkdir(parents=True, exist_ok=True)
        validated_dir.mkdir(parents=True, exist_ok=True)
        profiles_dir.mkdir(parents=True, exist_ok=True)
        
        return str(tmp_path), "test_dataset"
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe with various data quality issues"""
        return pd.DataFrame({
            'ID': [1, 2, 3, 4, 5, 5],  # Has duplicate
            'Name': ['  Alice  ', 'Bob', None, 'David', 'Eve', 'Eve'],
            'Age': [25, None, 35, None, 40, 40],
            'Score': [85.5, 90.0, 9999.0, 92.0, 0.01, 0.01],  # Has outliers
            'Category': ['A', 'B', 'A', 'B', 'A', 'A'],
            'Date': ['2020-01-01', '2020-02-01', None, '2020-04-01', '2020-05-01', '2020-05-01'],
            'Constant': [1, 1, 1, 1, 1, 1]  # Constant column
        })
    
    def test_normalize_column_names(self, setup_test_env, sample_dataframe):
        """Test that column names are normalized correctly"""
        project_root, dataset_name = setup_test_env
        
        # Create a cleaner instance
        cleaner = DataCleaner(dataset_name)
        cleaner.schema_profile = None
        cleaner.config = {
            'cleaning': {'outlier_method': 'iqr', 'outlier_threshold': 3}
        }
        
        # Test normalization
        df_normalized = cleaner.normalize_column_names(sample_dataframe.copy())
        
        # Check that column names are lowercase
        assert all(col.islower() or '_' in col for col in df_normalized.columns)
        # Check that all columns use underscores
        assert all(' ' not in col for col in df_normalized.columns)
    
    def test_handle_missing_values_categorical(self, setup_test_env):
        """Test that missing values in categorical columns are filled"""
        project_root, dataset_name = setup_test_env
        
        cleaner = DataCleaner(dataset_name)
        cleaner.schema_profile = {
            'column_types': {
                'categorical': ['category']  # Use lowercase to match normalization
            }
        }
        cleaner.config = {
            'cleaning': {'outlier_method': 'iqr', 'outlier_threshold': 3}
        }
        
        df = pd.DataFrame({
            'category': ['A', 'B', None, 'A'],
            'value': [1, 2, 3, 4]
        })
        
        df_cleaned = cleaner.handle_missing_values(df)
        
        # Check that there are no nulls in category
        assert df_cleaned['category'].isnull().sum() == 0
    
    def test_handle_missing_values_numeric(self, setup_test_env):
        """Test that missing values in numeric columns are filled with median"""
        project_root, dataset_name = setup_test_env
        
        cleaner = DataCleaner(dataset_name)
        cleaner.schema_profile = {
            'column_types': {
                'continuous_numeric': ['score']  # Use lowercase to match normalization
            }
        }
        cleaner.config = {
            'cleaning': {'outlier_method': 'iqr', 'outlier_threshold': 3}
        }
        
        df = pd.DataFrame({
            'score': [10.0, 20.0, None, 30.0, 40.0]
        })
        
        df_cleaned = cleaner.handle_missing_values(df)
        
        # Check that there are no nulls in score
        assert df_cleaned['score'].isnull().sum() == 0
        # Check that missing value was filled with median (25.0 - average of 20.0 and 30.0)
        assert df_cleaned['score'].iloc[2] == 25.0
    
    def test_remove_duplicates(self, setup_test_env):
        """Test that duplicate rows are removed"""
        project_root, dataset_name = setup_test_env
        
        cleaner = DataCleaner(dataset_name)
        cleaner.config = {
            'cleaning': {'outlier_method': 'iqr', 'outlier_threshold': 3}
        }
        
        df = pd.DataFrame({
            'ID': [1, 2, 3, 3],
            'Value': [10, 20, 30, 30]
        })
        
        initial_rows = len(df)
        df_cleaned = cleaner.remove_duplicates(df)
        
        # Check that one duplicate was removed
        assert len(df_cleaned) == initial_rows - 1
        assert df_cleaned['ID'].nunique() == len(df_cleaned)
    
    def test_handle_outliers_iqr_method(self, setup_test_env):
        """Test that outliers are capped using IQR method"""
        project_root, dataset_name = setup_test_env
        
        cleaner = DataCleaner(dataset_name)
        cleaner.schema_profile = {
            'column_types': {
                'continuous_numeric': ['value']  # Use lowercase to match normalization
            }
        }
        cleaner.config = {
            'cleaning': {'outlier_method': 'iqr', 'outlier_threshold': 3}
        }
        
        df = pd.DataFrame({
            'value': [10, 12, 14, 16, 18, 20, 1000]  # 1000 is an outlier
        })
        
        df_cleaned = cleaner.handle_outliers(df)
        
        # Check that the outlier (1000) has been capped
        assert df_cleaned['value'].max() < 1000
        # Check that operation was logged
        assert len(cleaner.cleaning_metrics['operations']) > 0
    
    def test_remove_constant_columns(self, setup_test_env):
        """Test that constant columns are removed"""
        project_root, dataset_name = setup_test_env
        
        cleaner = DataCleaner(dataset_name)
        cleaner.config = {
            'cleaning': {'outlier_method': 'iqr', 'outlier_threshold': 3}
        }
        
        df = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'Value': [10, 20, 30, 40],
            'Constant': [5, 5, 5, 5]  # Constant column
        })
        
        initial_cols = len(df.columns)
        df_cleaned = cleaner.remove_constant_columns(df)
        
        # Check that constant column was removed
        assert len(df_cleaned.columns) == initial_cols - 1
        assert 'Constant' not in df_cleaned.columns
    
    def test_standardize_data_types(self, setup_test_env):
        """Test that data types are standardized correctly"""
        project_root, dataset_name = setup_test_env
        
        cleaner = DataCleaner(dataset_name)
        cleaner.schema_profile = {
            'column_types': {
                'datetime': ['date'],  # Use lowercase to match normalization
                'continuous_numeric': ['score'],
                'categorical': ['category']
            }
        }
        cleaner.config = {
            'cleaning': {'outlier_method': 'iqr', 'outlier_threshold': 3}
        }
        
        df = pd.DataFrame({
            'date': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'score': ['85.5', '90.0', '95.0'],
            'category': ['A', 'B', 'A']
        })
        
        df_cleaned = cleaner.standardize_data_types(df)
        
        # Check that date is converted to datetime
        assert pd.api.types.is_datetime64_any_dtype(df_cleaned['date'])
        # Check that score is converted to numeric
        assert pd.api.types.is_numeric_dtype(df_cleaned['score'])
        # Check that category is converted to category
        assert pd.api.types.is_categorical_dtype(df_cleaned['category'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

