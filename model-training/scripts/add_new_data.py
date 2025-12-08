# scripts/add_new_dataset.py
"""
Add New Dataset to DataCraft
Processes a new CSV and makes it available in the UI
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import ModelDataLoader
from feature_engineering import FeatureEngineer
from metadata_manager import MetadataManager

# Configuration
PROJECT_ID = "datacraft-data-pipeline"
BUCKET_NAME = "isha-retail-data"
DATASET_ID = "datacraft_ml"


def add_dataset(
    csv_path: str,
    dataset_name: str,
    upload_to_gcs: bool = True
):
    """
    Add a new dataset to DataCraft
    
    Args:
        csv_path: Path to CSV file
        dataset_name: Name for the dataset (e.g., 'customers', 'products')
        upload_to_gcs: Whether to upload to GCS
    
    Steps:
        1. Load CSV
        2. Upload to GCS (optional)
        3. Load to BigQuery
        4. Generate metadata
        5. Store metadata
    """
    
    print("=" * 70)
    print(f"ADDING NEW DATASET: {dataset_name}")
    print("=" * 70)
    
    # Step 1: Load CSV
    print(f"\n1. Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)[:5]}...")
    
    # Initialize loader
    loader = ModelDataLoader(BUCKET_NAME, PROJECT_ID, DATASET_ID)
    
    # Step 2: Upload to GCS (optional)
    if upload_to_gcs:
        print(f"\n2. Uploading to GCS...")
        from google.cloud import storage
        
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(BUCKET_NAME)
        
        # Upload to validated folder
        blob_path = f"data/validated/{dataset_name}_validated.csv"
        blob = bucket.blob(blob_path)
        
        df.to_csv(f'/tmp/{dataset_name}_validated.csv', index=False)
        blob.upload_from_filename(f'/tmp/{dataset_name}_validated.csv')
        
        print(f"   ✓ Uploaded to: gs://{BUCKET_NAME}/{blob_path}")
    
    # Step 3: Load to BigQuery
    print(f"\n3. Loading to BigQuery...")
    table_id = loader.load_to_bigquery(df, dataset_name, table_suffix="_processed")
    print(f"   ✓ Created table: {table_id}")
    
    # Step 4: Generate metadata
    print(f"\n4. Generating metadata...")
    engineer = FeatureEngineer(df, {}, dataset_name)
    
    metadata = engineer.generate_metadata()
    llm_context = engineer.create_llm_context()
    
    print(f"   ✓ Generated metadata for {len(metadata['columns'])} columns")
    print(f"   ✓ Created LLM context ({len(llm_context)} chars)")
    
    # Step 5: Store metadata
    print(f"\n5. Storing metadata in BigQuery...")
    manager = MetadataManager(PROJECT_ID, DATASET_ID)
    manager.store_metadata(dataset_name, metadata, llm_context)
    
    print(f"   ✓ Metadata stored")
    
    # Verify
    print(f"\n6. Verifying...")
    retrieved = manager.get_metadata(dataset_name)
    
    if retrieved:
        print(f"   ✓ Verification successful!")
        print(f"\n" + "=" * 70)
        print(f"✅ DATASET '{dataset_name}' ADDED SUCCESSFULLY!")
        print(f"=" * 70)
        print(f"\nDataset is now available in:")
        print(f"  • BigQuery: {table_id}")
        print(f"  • DataCraft UI: Will appear in dataset selector")
        print(f"  • Rows: {len(df):,}")
        print(f"  • Columns: {len(df.columns)}")
        print("=" * 70)
    else:
        print(f"   ✗ Verification failed!")


if __name__ == "__main__":
    """
    Usage:
        python add_new_dataset.py
    
    Then follow prompts
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Add new dataset to DataCraft')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--name', type=str, help='Dataset name (e.g., customers)')
    parser.add_argument('--no-upload', action='store_true', help='Skip GCS upload')
    
    args = parser.parse_args()
    
    # Interactive mode
    if not args.csv:
        print("\n" + "=" * 70)
        print("ADD NEW DATASET TO DATACRAFT")
        print("=" * 70 + "\n")
        
        csv_path = input("Enter path to CSV file: ").strip()
        dataset_name = input("Enter dataset name (e.g., customers): ").strip().lower()
        upload = input("Upload to GCS? (y/n): ").strip().lower() == 'y'
    else:
        csv_path = args.csv
        dataset_name = args.name
        upload = not args.no_upload
    
    # Validate inputs
    if not Path(csv_path).exists():
        print(f"✗ File not found: {csv_path}")
        sys.exit(1)
    
    if not dataset_name:
        print("✗ Dataset name required")
        sys.exit(1)
    
    # Process
    try:
        add_dataset(csv_path, dataset_name, upload)
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)