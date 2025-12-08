# app.py
"""
DataCraft - Interactive Data Insights Platform
Streamlit UI for natural language data querying
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
from pathlib import Path
import os
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from query_handler import QueryHandler
from visualization_engine import VisualizationEngine
from dataset_manager import DatasetManager
from dataset_describer import DatasetDescriber

# Page configuration
st.set_page_config(
    page_title="DataCraft - Data Insights Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = 'orders'
if 'dataset_description' not in st.session_state:
    st.session_state.dataset_description = None
if 'sample_queries' not in st.session_state:
    st.session_state.sample_queries = []
if 'show_description' not in st.session_state:
    st.session_state.show_description = True
if 'handler' not in st.session_state:
    # Load configuration from environment or defaults
    config = {
        'project_id': os.getenv('GCP_PROJECT_ID', 'datacraft-data-pipeline'),
        'dataset_id': os.getenv('BQ_DATASET', 'datacraft_ml'),
        'bucket_name': os.getenv('GCS_BUCKET_NAME', 'isha-retail-data'),
        'region': os.getenv('GCP_REGION', 'us-central1'),
        'model_name': os.getenv('BEST_MODEL_NAME', 'gemini-2.5-pro'),
        'table_name': 'orders_processed'
    }
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

# Header
st.markdown('<div class="main-header">📊 DataCraft</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your data in plain English</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Dataset selector
    st.subheader("📊 Select Dataset")
    
    available_datasets = st.session_state.dataset_manager.get_available_datasets()
    
    if available_datasets:
        dataset_options = [ds['name'] for ds in available_datasets]
        dataset_labels = [
            f"{ds['name']} ({ds['rows']:,} rows)" for ds in available_datasets
        ]
        
        selected_idx = st.selectbox(
            "Choose dataset:",
            range(len(dataset_options)),
            format_func=lambda i: dataset_labels[i],
            key='dataset_selector'
        )
        
        dataset_name = dataset_options[selected_idx]
        
        # Check if dataset changed
        if st.session_state.selected_dataset != dataset_name:
            st.session_state.selected_dataset = dataset_name
            st.session_state.dataset_description = None
            st.session_state.sample_queries = []
            st.session_state.show_description = True
        
        st.session_state.handler.table_name = f"{dataset_name}_processed"
        
    else:
        st.warning("No datasets found. Run the data pipeline first.")
        dataset_name = 'orders'
    
    # Upload new dataset
    st.subheader("⬆️ Upload New Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    new_dataset_name = st.text_input("Dataset Name (no spaces, lowercase)", value="")

    if uploaded_file and new_dataset_name and st.button("📤 Upload & Register Dataset"):
        with st.spinner("Processing dataset..."):
            result = st.session_state.dataset_manager.upload_dataset_from_ui(
                uploaded_file,
                new_dataset_name.strip().lower().replace(" ", "_")
            )

            if result["status"] == "success":
                st.success(f"✅ Dataset '{new_dataset_name}' uploaded successfully!")
                st.rerun()
            else:
                st.error(f"❌ Upload failed: {result['error']}")

    st.markdown("---")

    # Model info
    st.subheader("🤖 AI Model")
    st.info(f"**Model:** {st.session_state.handler.model_name}\n\n**Status:** ✅ Ready")
    
    # Dataset stats
    st.subheader("📋 Quick Stats")
    dataset_info = st.session_state.dataset_manager.get_dataset_info(dataset_name)
    if dataset_info:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", f"{dataset_info['row_count']:,}")
        with col2:
            st.metric("Columns", dataset_info['column_count'])

    with st.expander("📑 View All Columns", expanded=False):
        columns = st.session_state.dataset_manager.get_dataset_columns(dataset_name)
        
    st.markdown("---")
    
    # Dynamic sample queries from LLM
    st.subheader("💡 Sample Queries")
    
    # Generate sample queries if not cached
    if not st.session_state.sample_queries or st.session_state.sample_queries[0].get('dataset') != dataset_name:
        with st.spinner("Generating sample queries..."):
            queries = st.session_state.dataset_describer.generate_sample_queries(dataset_name)
            st.session_state.sample_queries = [{'query': q, 'dataset': dataset_name} for q in queries]
    
    # Display sample queries as buttons
    for idx, query_data in enumerate(st.session_state.sample_queries):
        if st.button(query_data['query'], key=f"sample_{idx}", use_container_width=True):
            st.session_state.example_query = query_data['query']
            st.session_state.show_description = False
            st.rerun()

# Main content area
if st.session_state.show_description and not st.session_state.current_result:
    # Generate description if not cached
    if st.session_state.dataset_description is None or st.session_state.dataset_description.get('dataset_name') != dataset_name:
        with st.spinner("🤖 AI is analyzing your dataset..."):
            description_result = st.session_state.dataset_describer.generate_description(dataset_name)
            st.session_state.dataset_description = description_result
    
    # Display beautiful dataset description
    if st.session_state.dataset_description and st.session_state.dataset_description.get('status') == 'success':
        desc = st.session_state.dataset_description
        
        # Header section
        st.markdown(f"""
        <div class="dataset-header">
            <div class="dataset-title">📊 {dataset_name.title()} Dataset</div>
            <div class="dataset-overview">{desc['description']}</div>
        </div>
        """, unsafe_allow_html=True)
        
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
            quality = desc.get('data_quality', {}).get('completeness', 'Unknown')
            st.markdown(f"""
            <div class="metric-box" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <div class="metric-value">✓</div>
                <div class="metric-label">Data Quality</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Two column layout for insights
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Key Insights
            if desc.get('key_insights'):
                st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                st.markdown('<div class="insight-title">💡 Key Insights</div>', unsafe_allow_html=True)
                for insight in desc['key_insights']:
                    st.markdown(f'<div class="insight-item">• {insight}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col_right:
            # Suggested Analyses
            if desc.get('suggested_analyses'):
                st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                st.markdown('<div class="insight-title">📊 Suggested Analyses</div>', unsafe_allow_html=True)
                for analysis in desc['suggested_analyses']:
                    st.markdown(f'<div class="insight-item">• {analysis}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Data Quality section
        if desc.get('data_quality'):
            st.markdown('<div class="insight-card">', unsafe_allow_html=True)
            st.markdown('<div class="insight-title">✅ Data Quality Assessment</div>', unsafe_allow_html=True)
            quality = desc['data_quality']
            
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                completeness = quality.get('completeness', 'Unknown')
                quality_class = 'quality-good' if 'good' in completeness.lower() else ('quality-fair' if 'fair' in completeness.lower() else 'quality-poor')
                st.markdown(f'<span class="quality-badge {quality_class}">Completeness: {completeness}</span>', unsafe_allow_html=True)
            
            with col_q2:
                consistency = quality.get('consistency', 'Unknown')
                quality_class = 'quality-good' if 'good' in consistency.lower() else ('quality-fair' if 'fair' in consistency.lower() else 'quality-poor')
                st.markdown(f'<span class="quality-badge {quality_class}">Consistency: {consistency}</span>', unsafe_allow_html=True)
            
            if quality.get('notes'):
                st.markdown(f'<div class="insight-item" style="margin-top: 1rem;"><strong>Notes:</strong> {quality["notes"]}</div>', unsafe_allow_html=True)
            
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
    
    elif st.session_state.dataset_description and st.session_state.dataset_description.get('status') == 'error':
        st.error(f"❌ Failed to generate description: {st.session_state.dataset_description.get('error')}")

else:
    # Query interface
    st.subheader("🔍 Ask Your Question")
    
    # Query input
    user_query = st.text_input(
        "Enter your question:",
        value=st.session_state.get('example_query', ''),
        placeholder="e.g., What are the top 5 products by sales?",
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
    
    # Process query immediately after button definition
    if generate_button and user_query:
        st.session_state.show_description = False
        with st.spinner("🤖 Generating SQL and executing query..."):
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

# Display results (outside the else block)
if st.session_state.current_result and not st.session_state.show_description:
    result = st.session_state.current_result
    
    if result.get('status') == 'success':
        st.success("✅ Query processed successfully!")
        
        st.markdown("---")
        st.subheader("💬 Answer")
        
        answer = result.get('natural_language_answer', 'No answer generated')
        st.markdown(f"### {answer}")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "📋 Data", "🔍 SQL Query", "ℹ️ Details"])
        
        with tab1:
            viz_data = result.get('visualization_data')
            if viz_data is not None and not viz_data.empty:
                try:
                    fig = st.session_state.viz_engine.create_visualization(
                        data=viz_data,
                        viz_config=result['visualization'],
                        query=user_query
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
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No data available")
        
        with tab3:
            st.code(result.get('sql_query', ''), language='sql')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Execution Time", f"{result.get('execution_time', 0):.2f}s")
            with col2:
                st.metric("Rows Returned", result.get('result_count', 0))
            with col3:
                status = "✅ Valid" if result.get('results_valid', False) else "⚠️ Invalid"
                st.metric("Validation", status)
        
        with tab4:
            st.subheader("Explanation")
            st.write(result.get('explanation', 'No explanation available'))
            
            st.subheader("Visualization Config")
            st.json(result.get('visualization', {}))
    
    elif result.get('status') == 'error':
        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

# Query history in sidebar
if st.session_state.query_history:
    with st.sidebar:
        st.markdown("---")
        st.subheader("📜 Recent Queries")
        
        for idx, hist in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(f"{hist['query'][:40]}...", expanded=False):
                st.write(f"**Time:** {hist['timestamp'][:19]}")
                st.write(f"**Status:** {hist['status']}")
                st.write(f"**Duration:** {hist['execution_time']:.2f}s")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🤖 Powered by Google Gemini")
with col2:
    st.caption("☁️ Running on Google Cloud")
with col3:
    st.caption("🔒 Secure & Validated")