"""
Unit tests for utility functions
Tests helper functions like encoding detection, formatting, validation
"""

import pytest
import tempfile
from pathlib import Path
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from utils import (detect_encoding, format_size, format_duration, 
                   ensure_dir, validate_config, get_dataset_path)


class TestUtils:
    """Test suite for utility functions"""
    
    def test_detect_encoding_utf8(self, tmp_path):
        """Test encoding detection for UTF-8 files"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World 你好", encoding='utf-8')
        
        encoding, confidence = detect_encoding(str(test_file))
        
        assert encoding is not None
        assert confidence > 0
    
    def test_format_size_bytes(self):
        """Test size formatting for bytes"""
        assert format_size(512) == "512.00 B"
    
    def test_format_size_kb(self):
        """Test size formatting for KB"""
        assert "KB" in format_size(2048)
    
    def test_format_size_mb(self):
        """Test size formatting for MB"""
        assert "MB" in format_size(1048576)
    
    def test_format_size_gb(self):
        """Test size formatting for GB"""
        assert "GB" in format_size(1073741824)
    
    def test_format_duration_seconds(self):
        """Test duration formatting for seconds"""
        assert format_duration(30) == "30.0s"
    
    def test_format_duration_minutes(self):
        """Test duration formatting for minutes"""
        result = format_duration(125)
        assert "m" in result and "s" in result
    
    def test_format_duration_hours(self):
        """Test duration formatting for hours"""
        result = format_duration(7265)
        assert "h" in result
    
    def test_ensure_dir_creates_directory(self, tmp_path):
        """Test that ensure_dir creates directories"""
        test_dir = tmp_path / "test" / "nested" / "path"
        
        ensure_dir(str(test_dir))
        
        assert test_dir.exists() and test_dir.is_dir()
    
    def test_ensure_dir_handles_existing(self, tmp_path):
        """Test that ensure_dir handles existing directories"""
        test_dir = tmp_path / "existing_dir"
        test_dir.mkdir()
        
        # Should not raise error
        ensure_dir(str(test_dir))
        
        assert test_dir.exists()
    
    def test_validate_config_valid(self):
        """Test that valid configuration passes validation"""
        valid_config = {
            'data': {
                'raw_path': 'data/raw',
                'processed_path': 'data/processed',
                'validated_path': 'data/validated'
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
                'auto_detect_protected_attrs': True
            },
            'cleaning': {
                'auto_detect_types': True
            }
        }
        
        is_valid, errors = validate_config(valid_config)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_config_missing_sections(self):
        """Test that missing configuration sections are caught"""
        incomplete_config = {
            'data': {
                'raw_path': 'data/raw'
            }
        }
        
        is_valid, errors = validate_config(incomplete_config)
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_validate_config_invalid_thresholds(self):
        """Test that invalid threshold values are caught"""
        invalid_config = {
            'data': {
                'raw_path': 'data/raw',
                'processed_path': 'data/processed',
                'validated_path': 'data/validated'
            },
            'gcp': {
                'bucket_name': 'test-bucket',
                'project_id': 'test-project'
            },
            'validation': {
                'null_threshold': 1.5,  # Should be between 0 and 1
                'duplicate_threshold': 0.05,
                'min_rows': 10,
                'min_columns': 2
            },
            'bias': {},
            'cleaning': {}
        }
        
        is_valid, errors = validate_config(invalid_config)
        
        assert is_valid is False
        assert any('null_threshold' in str(error) for error in errors)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

