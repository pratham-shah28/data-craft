# app.py - Complete Integration with Beautiful Unstructured Data UI
"""
DataCraft - Interactive Data Insights Platform
Unified UI for both structured and unstructured data
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import os
from datetime import datetime
import json
import tempfile

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from query_handler import QueryHandler
from visualization_engine import VisualizationEngine
from dataset_manager import DatasetManager
from dataset_describer import DatasetDescriber
from unstructured_data_handler import UnstructuredDataHandler

# Page configuration
st.set_page_config(
    page_title="DataCraft - Data Insights Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 1rem;
    }
    .dataset-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .unstructured-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .dataset-title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .dataset-overview {
        font-size: 1.1rem;
        line-height: 1.6;
        opacity: 0.95;
    }
    .insight-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .pdf-card {
        background: white;
        padding: 1rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #f5576c;
    }
    .insight-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.8rem;
    }
    .insight-item {
        padding: 0.6rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .insight-item:last-child {
        border-bottom: none;
    }
    .metric-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .quality-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .quality-good {
        background: #d4edda;
        color: #155724;
    }
    .quality-fair {
        background: #fff3cd;
        color: #856404;
    }
    .quality-poor {
        background: #f8d7da;
        color: #721c24;
    }
    .json-preview {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .pdf-icon {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
if 'dataset_type' not in st.session_state:
    st.session_state.dataset_type = 'structured'  # 'structured' or 'unstructured'
if 'dataset_description' not in st.session_state:
    st.session_state.dataset_description = None
if 'sample_queries' not in st.session_state:
    st.session_state.sample_queries = []
if 'show_description' not in st.session_state:
    st.session_state.show_description = True
if 'uploaded_pdfs' not in st.session_state:
    st.session_state.uploaded_pdfs = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'handler' not in st.session_state:
    # Load configuration
    config = {
        'project_id': os.getenv('GCP_PROJECT_ID', 'datacraft-data-pipeline'),
        'dataset_id': os.getenv('BQ_DATASET', 'datacraft_ml'),
        'bucket_name': os.getenv('GCS_BUCKET_NAME', 'isha-retail-data'),
        'region': os.getenv('GCP_REGION', 'us-central1'),
        'model_name': os.getenv('BEST_MODEL_NAME', 'gemini-2.5-pro'),
        'table_name': 'orders_processed'
    }
    st.session_state.config = config
    st.session_state.handler = QueryHandler(config)
    st.session_state.viz_engine = VisualizationEngine()
    st.session_state.dataset_manager = DatasetManager(
        config['project_id'],
        config['dataset_id'],
        config['bucket_name']
    )
    st.session_state.dataset_describer = DatasetDescriber(
        config['project_id'],
        config['dataset_id'],
        config['region'],
        config['model_name']
    )
    st.session_state.unstructured_handler = UnstructuredDataHandler(
        config['project_id'],
        config['dataset_id'],
        config['region']
    )

# Header
st.markdown('<div class="main-header">📊 DataCraft</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your data in plain English</div>', unsafe_allow_html=True)

def render_structured_sidebar():
    """Render sidebar for structured data"""
    st.subheader("📊 Select Dataset")
    
    available_datasets = st.session_state.dataset_manager.get_available_datasets()
    
    if available_datasets:
        # Filter out unstructured datasets (those with _metadata column)
        structured_datasets = [ds for ds in available_datasets if not ds['name'].endswith('_unstructured')]
        
        if structured_datasets:
            dataset_options = [ds['name'] for ds in structured_datasets]
            dataset_labels = [f"{ds['name']} ({ds['rows']:,} rows)" for ds in structured_datasets]
            
            selected_idx = st.selectbox(
                "Choose dataset:",
                range(len(dataset_options)),
                format_func=lambda i: dataset_labels[i],
                key='structured_dataset_selector'
            )
            
            dataset_name = dataset_options[selected_idx]
            
            if st.session_state.selected_dataset != dataset_name or st.session_state.dataset_type != 'structured':
                st.session_state.selected_dataset = dataset_name
                st.session_state.dataset_type = 'structured'
                st.session_state.dataset_description = None
                st.session_state.sample_queries = []
                st.session_state.show_description = True
                st.session_state.current_result = None
                st.rerun()
            
            st.session_state.handler.table_name = f"{dataset_name}_processed"
        else:
            st.info("No structured datasets found")
    
    st.markdown("---")
    
    # Upload CSV
    st.subheader("⬆️ Upload CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv_uploader")
    new_dataset_name = st.text_input("Dataset Name", key="csv_name")

    if uploaded_file and new_dataset_name and st.button("📤 Upload", key="upload_csv"):
        with st.spinner("Processing..."):
            result = st.session_state.dataset_manager.upload_dataset_from_ui(
                uploaded_file,
                new_dataset_name.strip().lower().replace(" ", "_")
            )
            if result["status"] == "success":
                st.success("✅ Uploaded!")
                st.rerun()
            else:
                st.error(f"❌ {result['error']}")
    
    # Show stats if dataset selected
    if st.session_state.dataset_type == 'structured' and st.session_state.selected_dataset:
        st.markdown("---")
        show_structured_stats()


def render_unstructured_sidebar():
    """Render sidebar for unstructured data"""
    st.subheader("📄 Unstructured Data")
    
    # Show existing unstructured datasets
    unstructured_tables = st.session_state.unstructured_handler.list_unstructured_tables()
    
    if unstructured_tables:
        st.markdown("**Existing Datasets:**")
        for table in unstructured_tables:
            if st.button(
                f"📄 {table['name']} ({table['rows']} docs)",
                key=f"select_{table['name']}",
                use_container_width=True
            ):
                st.session_state.selected_dataset = table['name']
                st.session_state.dataset_type = 'unstructured'
                st.session_state.processing_complete = True
                st.session_state.show_description = True
                st.session_state.handler.table_name = f"{table['name']}_processed"
                st.rerun()
    
    st.markdown("---")
    
    # Upload PDFs
    st.subheader("⬆️ Upload PDFs")
    
    doc_type = st.text_input(
        "Document Type",
        value="invoices",
        help="e.g., invoices, receipts, insurance",
        key="doc_type_input"
    )
    
    examples_dir = st.text_input(
        "Examples Path (Optional)",
        value="",
        help="Path to example PDFs with JSONs",
        key="examples_dir_input"
    )
    
    uploaded_files = st.file_uploader(
        "Select PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    if uploaded_files:
        st.session_state.uploaded_pdfs = uploaded_files
        st.session_state.doc_type = doc_type
        st.session_state.examples_dir = examples_dir if examples_dir else None
        st.session_state.dataset_type = 'unstructured'
        st.session_state.processing_complete = False
        st.success(f"✅ {len(uploaded_files)} file(s) ready")
        
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            process_unstructured_files()


def show_structured_stats():
    """Show stats for selected structured dataset"""
    dataset_name = st.session_state.selected_dataset
    dataset_info = st.session_state.dataset_manager.get_dataset_info(dataset_name)
    
    if dataset_info:
        st.subheader("📋 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", f"{dataset_info['row_count']:,}")
        with col2:
            st.metric("Columns", dataset_info['column_count'])
        
        with st.expander("📑 View Columns", expanded=False):
            columns = st.session_state.dataset_manager.get_dataset_columns(dataset_name)
            if columns:
                mid = len(columns) // 2
                col_left, col_right = st.columns(2)
                with col_left:
                    for col in columns[:mid]:
                        st.markdown(f"• `{col}`")
                with col_right:
                    for col in columns[mid:]:
                        st.markdown(f"• `{col}`")
        
        st.markdown("---")
        
        # Sample queries
        st.subheader("💡 Sample Queries")
        if not st.session_state.sample_queries or st.session_state.sample_queries[0].get('dataset') != dataset_name:
            with st.spinner("Generating..."):
                queries = st.session_state.dataset_describer.generate_sample_queries(dataset_name)
                st.session_state.sample_queries = [{'query': q, 'dataset': dataset_name} for q in queries]
        
        for idx, query_data in enumerate(st.session_state.sample_queries):
            if st.button(query_data['query'], key=f"sample_{idx}", use_container_width=True):
                st.session_state.example_query = query_data['query']
                st.session_state.show_description = False
                st.rerun()


def render_welcome_screen():
    """Show welcome screen when no dataset selected"""
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">👋 Welcome to DataCraft</h1>
        <p style="font-size: 1.3rem; color: #666; margin-bottom: 2rem;">
            Your AI-powered data insights platform
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 3rem;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 1rem; width: 300px;">
                <h3>📊 Structured Data</h3>
                <p>Upload CSV files or select existing datasets to query with natural language</p>
            </div>
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 2rem; border-radius: 1rem; width: 300px;">
                <h3>📄 Unstructured Data</h3>
                <p>Extract data from PDFs, invoices, receipts using AI and query them naturally</p>
            </div>
        </div>
        <p style="margin-top: 3rem; color: #999;">Select a data type from the sidebar to get started</p>
    </div>
    """, unsafe_allow_html=True)


def render_unstructured_upload_preview():
    """Show preview of uploaded PDFs before processing"""
    st.markdown(f"""
    <div class="unstructured-header">
        <div class="dataset-title">📄 Document Upload Preview</div>
        <div class="dataset-overview">Ready to process {len(st.session_state.uploaded_pdfs)} PDF document(s)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Document type and count
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-value">{len(st.session_state.uploaded_pdfs)}</div>
            <div class="metric-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_size = sum(f.size for f in st.session_state.uploaded_pdfs) / (1024 * 1024)
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-value">{total_size:.1f} MB</div>
            <div class="metric-label">Total Size</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <div class="metric-value">{st.session_state.doc_type}</div>
            <div class="metric-label">Document Type</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # File list
    st.subheader("📋 Files to Process")
    for idx, file in enumerate(st.session_state.uploaded_pdfs, 1):
        st.markdown(f"""
        <div class="pdf-card">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div class="pdf-icon">📄</div>
                <div style="flex: 1;">
                    <strong>{file.name}</strong><br>
                    <small style="color: #666;">{file.size / 1024:.1f} KB</small>
                </div>
                <div style="background: #28a745; color: white; padding: 0.3rem 0.8rem; border-radius: 1rem; font-size: 0.85rem;">
                    Ready
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Info about processing
    st.info("""
    **What happens next:**
    1. 🤖 AI will extract structured data from each PDF
    2. 💾 Data will be stored in BigQuery table: `{}_processed`
    3. 🔍 You'll be able to query the data naturally
    """.format(st.session_state.doc_type))


def process_unstructured_files():
    """Process uploaded PDF files"""
    with st.spinner("🤖 Extracting data from documents..."):
        try:
            temp_paths = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in st.session_state.uploaded_pdfs:
                    temp_path = os.path.join(temp_dir, file.name)
                    with open(temp_path, 'wb') as f:
                        f.write(file.read())
                    temp_paths.append(temp_path)
                
                extracted_data = st.session_state.unstructured_handler.process_multiple_files(
                    temp_paths,
                    st.session_state.doc_type,
                    st.session_state.examples_dir
                )
            
            st.session_state.extracted_data = extracted_data
            
            # Store in BigQuery
            with st.spinner("💾 Storing in BigQuery..."):
                table_id = st.session_state.unstructured_handler.store_in_bigquery(
                    extracted_data,
                    st.session_state.doc_type
                )
            
            st.session_state.processing_complete = True
            st.session_state.selected_dataset = st.session_state.doc_type
            st.session_state.handler.table_name = f"{st.session_state.doc_type}_processed"
            st.success(f"✅ Processed {len(extracted_data)} documents!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")


def render_unstructured_overview():
    """Show overview of processed unstructured data"""
    dataset_name = st.session_state.selected_dataset
    
    st.markdown(f"""
    <div class="unstructured-header">
        <div class="dataset-title">📄 {dataset_name.title()} Documents</div>
        <div class="dataset-overview">Extracted and structured data from {len(st.session_state.extracted_data) if st.session_state.extracted_data else 'your'} PDF documents</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get table info
    table_info = st.session_state.unstructured_handler.get_table_info(dataset_name)
    
    if table_info:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="metric-value">{table_info['num_rows']}</div>
                <div class="metric-label">Documents</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="metric-value">{table_info['num_columns']}</div>
                <div class="metric-label">Fields Extracted</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="metric-value">{round(table_info['size_bytes'] / (1024 * 1024), 2)} MB</div>
                <div class="metric-label">Data Size</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <div class="metric-value">✓</div>
                <div class="metric-label">Ready to Query</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show extracted data preview
    if st.session_state.extracted_data:
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown('<div class="insight-title">📄 Extracted Data Preview</div>', unsafe_allow_html=True)
            
            for i, data in enumerate(st.session_state.extracted_data[:3], 1):
                with st.expander(f"Document {i}", expanded=(i==1)):
                    # Remove _metadata for cleaner display
                    display_data = {k: v for k, v in data.items() if k != '_metadata'}
                    st.json(display_data)
            
            if len(st.session_state.extracted_data) > 3:
                st.info(f"... and {len(st.session_state.extracted_data) - 3} more documents")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_right:
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown('<div class="insight-title">📊 BigQuery Storage</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            **Table ID:** `{table_info['table_id'] if table_info else 'N/A'}`
            
            **Schema:**
            - Each document = 1 row
            - Fields auto-detected from PDFs
            - Nested data stored as JSON
            - `_metadata` column with extraction info
            
            **Query Examples:**
            - "What is the total amount from all {dataset_name}?"
            - "Show me {dataset_name} from last month"
            - "Which vendor has the most {dataset_name}?"
            """)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("🔄 Upload More", use_container_width=True):
            st.session_state.uploaded_pdfs = []
            st.session_state.extracted_data = []
            st.session_state.processing_complete = False
            st.rerun()
    
    with col_btn2:
        if st.button("❓ Start Querying", type="primary", use_container_width=True):
            st.session_state.show_description = False
            st.rerun()


def render_structured_overview():
    """Render structured dataset overview (existing code)"""
    dataset_name = st.session_state.selected_dataset
    
    # Generate description if needed
    if st.session_state.dataset_description is None or st.session_state.dataset_description.get('dataset_name') != dataset_name:
        with st.spinner("🤖 AI is analyzing your dataset..."):
            description_result = st.session_state.dataset_describer.generate_description(dataset_name)
            st.session_state.dataset_description = description_result
    
    # Display (same as before)
    if st.session_state.dataset_description and st.session_state.dataset_description.get('status') == 'success':
        desc = st.session_state.dataset_description
        
        st.markdown(f"""
        <div class="dataset-header">
            <div class="dataset-title">📊 {dataset_name.title()} Dataset</div>
            <div class="dataset-overview">{desc['description']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics and insights (same as before)
        render_structured_metrics_and_insights(desc)


def render_structured_metrics_and_insights(desc):
    """Render metrics and insights for structured data"""
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <div class="metric-value">{desc['metadata']['row_count']:,}</div>
            <div class="metric-label">Total Rows</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-value">{desc['metadata']['column_count']}</div>
            <div class="metric-label">Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-value">{desc['metadata']['size_mb']} MB</div>
            <div class="metric-label">Dataset Size</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <div class="metric-value">✓</div>
            <div class="metric-label">Data Quality</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Insights
    col_left, col_right = st.columns(2)
    
    with col_left:
        if desc.get('key_insights'):
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown('<div class="insight-title">💡 Key Insights</div>', unsafe_allow_html=True)
            for insight in desc['key_insights']:
                st.markdown(f'<div class="insight-item">• {insight}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        if desc.get('suggested_analyses'):
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown('<div class="insight-title">📊 Suggested Analyses</div>', unsafe_allow_html=True)
            for analysis in desc['suggested_analyses']:
                st.markdown(f'<div class="insight-item">• {analysis}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("🔄 Regenerate Description", use_container_width=True):
            st.session_state.dataset_description = None
            st.rerun()
    
    with col_btn2:
        if st.button("❓ Start Querying", type="primary", use_container_width=True):
            st.session_state.show_description = False
            st.rerun()


def render_query_interface():
    """Render query interface (works for both structured and unstructured)"""
    dataset_name = st.session_state.selected_dataset
    dataset_type = "📄 Documents" if st.session_state.dataset_type == 'unstructured' else "📊 Dataset"
    
    st.subheader(f"🔍 Query: {dataset_name.title()} {dataset_type}")
    
    user_query = st.text_input(
        "Ask your question:",
        value=st.session_state.get('example_query', ''),
        placeholder=f"e.g., What is the total amount from all {dataset_name}?",
        key="query_input"
    )
    
    col_a, col_b, col_c = st.columns([1, 1, 4])
    
    with col_a:
        generate_button = st.button("🚀 Generate", type="primary", use_container_width=True)
    
    with col_b:
        if st.button("⬅️ Back to Overview", use_container_width=True):
            st.session_state.show_description = True
            st.session_state.current_result = None
            st.session_state.example_query = ''
            st.rerun()
    
    if generate_button and user_query:
        st.session_state.show_description = False
        with st.spinner("🤖 Processing query..."):
            try:
                result = st.session_state.handler.process_query(user_query, dataset_name)
                st.session_state.current_result = result
                st.session_state.query_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'query': user_query,
                    'status': result.get('status', 'unknown'),
                    'execution_time': result.get('execution_time', 0)
                })
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.current_result = {'status': 'error', 'error': str(e)}

    # Display results
    if st.session_state.current_result and not st.session_state.show_description:
        display_query_results()


def display_query_results():
    """Display query results (same for both data types)"""
    result = st.session_state.current_result
    
    if result.get('status') == 'success':
        st.success("✅ Query executed successfully!")
        
        st.markdown("---")
        st.subheader("💬 Answer")
        answer = result.get('natural_language_answer', 'No answer')
        st.markdown(f"### {answer}")
        
        tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📋 Data", "🔍 SQL Query"])
        
        with tab1:
            viz_data = result.get('visualization_data')
            if viz_data is not None and not viz_data.empty:
                try:
                    fig = st.session_state.viz_engine.create_visualization(
                        data=viz_data,
                        viz_config=result['visualization'],
                        query=st.session_state.get('example_query', '')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Using table view: {str(e)}")
                    st.dataframe(viz_data, use_container_width=True)
            else:
                st.info("No data to visualize")
        
        with tab2:
            viz_data = result.get('visualization_data')
            if viz_data is not None and not viz_data.empty:
                st.dataframe(viz_data, use_container_width=True, hide_index=True)
                csv = viz_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with tab3:
            st.code(result.get('sql_query', ''), language='sql')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Time", f"{result.get('execution_time', 0):.2f}s")
            with col2:
                st.metric("Rows", result.get('result_count', 0))
            with col3:
                status = "✅" if result.get('results_valid') else "⚠️"
                st.metric("Valid", status)
    
    elif result.get('status') == 'error':
        st.error(f"❌ {result.get('error', 'Unknown error')}")


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Data type selector
    data_type = st.radio(
        "Data Type",
        ["📊 Structured (CSV)", "📄 Unstructured (PDF)"],
        key="data_type_selector"
    )
    
    st.markdown("---")
    
    if data_type == "📊 Structured (CSV)":
        render_structured_sidebar()
    else:
        render_unstructured_sidebar()

# Main content
if st.session_state.dataset_type == 'structured' and st.session_state.selected_dataset:
    if st.session_state.show_description and not st.session_state.current_result:
        render_structured_overview()
    else:
        render_query_interface()
elif st.session_state.dataset_type == 'unstructured':
    if st.session_state.processing_complete:
        render_unstructured_overview()
    elif st.session_state.uploaded_pdfs:
        render_unstructured_upload_preview()
    else:
        render_welcome_screen()
else:
    render_welcome_screen()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🤖 Powered by Google Gemini")
with col2:
    st.caption("☁️ Running on Google Cloud")
with col3:
    st.caption("🔒 Secure & Validated")