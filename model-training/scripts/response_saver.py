# scripts/response_saver.py
"""
Response Saver for Best Model
Saves query responses from the selected best model to both local files and GCS
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path
import logging
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logging

from google.cloud import storage


class ResponseSaver:
    """
    Save model responses to organized directory structure
    Saves locally and uploads to GCS
    """
    
    def __init__(
        self,
        project_id: str,
        bucket_name: str,
        local_output_dir: str = "/opt/airflow/outputs/best-model-responses",
        service_account_path: Optional[str] = None
    ):
        """
        Initialize response saver
        
        Args:
            project_id: GCP project ID
            bucket_name: GCS bucket name for uploads
            local_output_dir: Local directory for saving responses
            service_account_path: Optional path to service account JSON
        """
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.local_output_dir = Path(local_output_dir)
        self.local_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(self.__class__.__name__)
        
        # Setup GCP credentials
        from utils import setup_gcp_credentials
        setup_gcp_credentials(service_account_path, self.logger)
        
        # Initialize GCS client
        try:
            self.storage_client = storage.Client(project=project_id)
            self.bucket = self.storage_client.bucket(bucket_name)
            self.logger.info(f"✓ GCS client initialized: gs://{bucket_name}")
        except Exception as e:
            self.logger.warning(f"Could not initialize GCS client: {e}")
            self.storage_client = None
            self.bucket = None
    
    def save_best_model_responses(
        self,
        model_name: str,
        responses: List[Dict],
        selection_report: Dict
    ) -> Dict:
        """
        Save all responses from the best model
        
        Args:
            model_name: Name of the best model
            responses: List of response dictionaries
            selection_report: Selection report with model scoring details
            
        Returns:
            Dictionary with save status and file paths
        """
        self.logger.info(f"Saving responses from best model: {model_name}")
        self.logger.info(f"Total responses: {len(responses)}")
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = self.local_output_dir / f"{timestamp}_{self._clean_model_name(model_name)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        # Save selection report first
        selection_file = run_dir / "model_selection_report.json"
        with open(selection_file, 'w') as f:
            json.dump(selection_report, f, indent=2)
        saved_files.append(str(selection_file))
        self.logger.info(f"✓ Saved selection report")
        
        # Save each query response
        for response in responses:
            query_num = response.get('query_number', 0)
            query_folder = run_dir / f"query_{query_num:03d}"
            
            files_saved = self._save_single_response(response, query_folder)
            saved_files.extend(files_saved)
        
        # Create summary file
        summary = self._create_summary(model_name, responses, selection_report)
        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files.append(str(summary_file))
        
        self.logger.info(f"✓ Saved {len(saved_files)} files locally")
        
        # Upload to GCS if available
        gcs_paths = []
        if self.bucket:
            gcs_paths = self._upload_to_gcs(run_dir, timestamp, model_name)
        
        return {
            "status": "success",
            "model_name": model_name,
            "local_directory": str(run_dir),
            "files_saved": len(saved_files),
            "gcs_uploaded": len(gcs_paths),
            "gcs_paths": gcs_paths[:10],  # First 10 for reference
            "timestamp": timestamp
        }
    
    def _save_single_response(
        self,
        response: Dict,
        output_folder: Path
    ) -> List[str]:
        """
        Save a single query response to organized files
        
        Args:
            response: Response dictionary
            output_folder: Folder to save files
            
        Returns:
            List of saved file paths
        """
        output_folder.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        try:
            # 1. Save user query
            query_file = output_folder / "user_query.txt"
            with open(query_file, 'w') as f:
                f.write(response.get('user_query', ''))
            saved_files.append(str(query_file))
            
            # 2. Save SQL query
            sql_file = output_folder / "sql_query.sql"
            with open(sql_file, 'w') as f:
                f.write(response.get('sql_query', ''))
            saved_files.append(str(sql_file))
            
            # 3. Save visualization config
            viz_file = output_folder / "visualization.json"
            with open(viz_file, 'w') as f:
                json.dump(response.get('visualization', {}), f, indent=2)
            saved_files.append(str(viz_file))
            
            # 4. Save explanation
            exp_file = output_folder / "explanation.txt"
            with open(exp_file, 'w') as f:
                f.write(response.get('explanation', ''))
            saved_files.append(str(exp_file))
            
            # 5. Save raw response (if available)
            if 'raw_response' in response:
                raw_file = output_folder / "raw_response.txt"
                with open(raw_file, 'w') as f:
                    f.write(response['raw_response'])
                saved_files.append(str(raw_file))
            
            # 6. Save complete result as JSON
            result_file = output_folder / "result.json"
            with open(result_file, 'w') as f:
                json.dump({
                    "query_number": response.get('query_number'),
                    "user_query": response.get('user_query'),
                    "sql_query": response.get('sql_query'),
                    "visualization": response.get('visualization'),
                    "explanation": response.get('explanation'),
                    "response_time": response.get('response_time'),
                    "status": response.get('status'),
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            saved_files.append(str(result_file))
            
        except Exception as e:
            self.logger.error(f"Error saving response: {e}")
        
        return saved_files
    
    def _create_summary(
        self,
        model_name: str,
        responses: List[Dict],
        selection_report: Dict
    ) -> Dict:
        """Create summary of saved responses"""
        successful = [r for r in responses if r.get('status') == 'success']
        failed = [r for r in responses if r.get('status') != 'success']
        
        return {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(responses),
            "successful_queries": len(successful),
            "failed_queries": len(failed),
            "model_selection_score": selection_report.get('best_model', {}).get('composite_score', 0),
            "model_performance_score": selection_report.get('best_model', {}).get('performance_score', 0),
            "model_bias_score": selection_report.get('best_model', {}).get('bias_score', 0),
            "queries": [
                {
                    "query_number": r.get('query_number'),
                    "user_query": r.get('user_query'),
                    "status": r.get('status'),
                    "sql_length": len(r.get('sql_query', '')),
                    "visualization_type": r.get('visualization', {}).get('type', 'unknown')
                }
                for r in responses
            ]
        }
    
    def _upload_to_gcs(
        self,
        local_dir: Path,
        timestamp: str,
        model_name: str
    ) -> List[str]:
        """
        Upload all files from local directory to GCS
        
        Args:
            local_dir: Local directory with files
            timestamp: Timestamp for GCS path
            model_name: Model name for GCS path
            
        Returns:
            List of GCS paths
        """
        if not self.bucket:
            self.logger.warning("GCS bucket not available, skipping upload")
            return []
        
        self.logger.info(f"Uploading to GCS: gs://{self.bucket_name}")
        
        clean_model = self._clean_model_name(model_name)
        gcs_base_path = f"best_model_responses/{timestamp}_{clean_model}"
        
        uploaded_paths = []
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = Path(root) / file
                relative_path = local_path.relative_to(local_dir)
                gcs_path = f"{gcs_base_path}/{relative_path}"
                
                try:
                    blob = self.bucket.blob(gcs_path)
                    blob.upload_from_filename(str(local_path))
                    uploaded_paths.append(f"gs://{self.bucket_name}/{gcs_path}")
                except Exception as e:
                    self.logger.error(f"Error uploading {local_path}: {e}")
        
        self.logger.info(f"✓ Uploaded {len(uploaded_paths)} files to GCS")
        
        return uploaded_paths
    
    def _clean_model_name(self, model_name: str) -> str:
        """Clean model name for use in file paths"""
        return model_name.replace('/', '_').replace('.', '_').replace(' ', '_')
    
    def save_metadata_for_deployment(
        self,
        model_name: str,
        selection_report: Dict,
        output_file: str = None
    ) -> str:
        """
        Save model metadata for deployment/production use
        
        Args:
            model_name: Best model name
            selection_report: Selection report
            output_file: Optional custom output file path
            
        Returns:
            Path to metadata file
        """
        if output_file is None:
            output_file = self.local_output_dir / "best_model_metadata.json"
        
        metadata = {
            "selected_model": model_name,
            "selection_date": datetime.now().isoformat(),
            "composite_score": selection_report.get('best_model', {}).get('composite_score', 0),
            "performance_score": selection_report.get('best_model', {}).get('performance_score', 0),
            "bias_score": selection_report.get('best_model', {}).get('bias_score', 0),
            "success_rate": selection_report.get('best_model', {}).get('success_rate', 0),
            "avg_response_time": selection_report.get('best_model', {}).get('response_time', 0),
            "deployment_ready": True,
            "selection_methodology": selection_report.get('selection_criteria', {})
        }
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✓ Saved deployment metadata: {output_file}")
        
        return str(output_file)


# ========================================
# STANDALONE TESTING
# ========================================

if __name__ == "__main__":
    """
    Test script
    Usage: python response_saver.py
    """
    
    # Mock responses
    mock_responses = [
        {
            "query_number": 1,
            "user_query": "What are top products?",
            "sql_query": "SELECT product_name, SUM(sales) FROM dataset GROUP BY product_name LIMIT 10",
            "visualization": {"type": "bar_chart", "title": "Top Products"},
            "explanation": "Shows top products by sales",
            "response_time": 1.5,
            "status": "success"
        },
        {
            "query_number": 2,
            "user_query": "Sales trend",
            "sql_query": "SELECT order_date, SUM(sales) FROM dataset GROUP BY order_date",
            "visualization": {"type": "line_chart", "title": "Sales Trend"},
            "explanation": "Time series analysis",
            "response_time": 1.2,
            "status": "success"
        }
    ]
    
    mock_selection_report = {
        "best_model": {
            "name": "gemini-1.5-pro",
            "composite_score": 92.5,
            "performance_score": 95.0,
            "bias_score": 15.0,
            "success_rate": 100.0,
            "response_time": 1.8
        },
        "selection_criteria": {
            "weights": {"performance": 0.5, "bias": 0.3, "speed": 0.1, "reliability": 0.1}
        }
    }
    
    print("\n" + "=" * 60)
    print("TESTING RESPONSE SAVER")
    print("=" * 60 + "\n")
    
    try:
        # Initialize saver
        print("1. Initializing response saver...")
        saver = ResponseSaver(
            project_id="datacraft-data-pipeline",
            bucket_name="isha-retail-data",
            local_output_dir="/tmp/best-model-responses"
        )
        print("   ✓ Saver initialized (GCS client optional for testing)")
        
        # Save responses
        print("\n2. Saving best model responses...")
        result = saver.save_best_model_responses(
            model_name="gemini-1.5-pro",
            responses=mock_responses,
            selection_report=mock_selection_report
        )
        
        print(f"\n3. Save Results:")
        print(f"   Status: {result['status']}")
        print(f"   Model: {result['model_name']}")
        print(f"   Local Directory: {result['local_directory']}")
        print(f"   Files Saved: {result['files_saved']}")
        
        # Save deployment metadata
        print("\n4. Saving deployment metadata...")
        metadata_file = saver.save_metadata_for_deployment(
            model_name="gemini-1.5-pro",
            selection_report=mock_selection_report
        )
        print(f"   ✓ Metadata saved: {metadata_file}")
        
        print("\n" + "=" * 60)
        print("✓ RESPONSE SAVER TEST PASSED!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)