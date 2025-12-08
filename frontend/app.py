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
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
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

# Header
st.markdown('<div class="main-header">📊 DataCraft</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your data in plain English</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Dataset selector
    dataset_name = st.selectbox(
        "Select Dataset",
        ["orders"],
        help="Choose which dataset to query"
    )
    
    # Model info
    st.subheader("🤖 AI Model")
    st.info(f"**Model:** {st.session_state.handler.model_name}\n\n**Status:** Ready")
    
    # Dataset info
    st.subheader("📊 Dataset Info")
    try:
        metadata = st.session_state.handler.get_dataset_metadata(dataset_name)
        if metadata:
            st.metric("Total Rows", f"{metadata.get('row_count', 0):,}")
            st.metric("Columns", metadata.get('column_count', 0))
    except:
        st.warning("Could not load dataset metadata")
    
    # Query suggestions
    st.subheader("💡 Example Queries")
    example_queries = [
        "What are the top 10 products by sales?",
        "Show me sales trend over the last year",
        "Which regions have the highest profit?",
        "Compare sales across customer segments",
        "What is the average discount by category?"
    ]
    
    for query in example_queries:
        if st.button(query, key=f"example_{query}", use_container_width=True):
            st.session_state.example_query = query

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Ask Your Question")
    
    # Query input
    user_query = st.text_input(
        "Enter your question:",
        value=st.session_state.get('example_query', ''),
        placeholder="e.g., What are total sales in 2024?",
        key="query_input"
    )
    
    col_a, col_b, col_c = st.columns([1, 1, 3])
    
    with col_a:
        submit_button = st.button("🚀 Generate", type="primary", use_container_width=True)
    
    with col_b:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

with col2:
    st.subheader("📈 Quick Stats")
    
    if st.session_state.query_history:
        st.metric("Queries Asked", len(st.session_state.query_history))
        successful = sum(1 for q in st.session_state.query_history if q.get('status') == 'success')
        st.metric("Successful", successful)
        st.metric("Success Rate", f"{successful/len(st.session_state.query_history)*100:.1f}%")
    else:
        st.info("No queries yet. Ask a question to get started!")

# Clear history
if clear_button:
    st.session_state.query_history = []
    st.session_state.current_result = None
    st.session_state.example_query = ''
    st.rerun()

# Process query
if submit_button and user_query:
    with st.spinner("🤖 Generating SQL and executing query..."):
        try:
            # Process query
            result = st.session_state.handler.process_query(user_query, dataset_name)
            
            # Store result
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
if st.session_state.current_result:
    result = st.session_state.current_result
    
    if result.get('status') == 'success':
        st.success("✅ Query processed successfully!")
        
        # Answer section
        st.markdown("---")
        st.subheader("💬 Answer")
        
        # Natural language answer
        answer = result.get('natural_language_answer', 'No answer generated')
        st.markdown(f"### {answer}")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "📋 Data", "🔍 SQL Query", "ℹ️ Details"])
        
        with tab1:
            # Visualization
            viz_data = result.get('visualization_data')
            if viz_data is not None and not viz_data.empty:
                try:
                    fig = st.session_state.viz_engine.create_visualization(
                        data=result['visualization_data'],
                        viz_config=result['visualization'],
                        query=user_query
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Visualization error: {e}")
                    st.dataframe(result['visualization_data'], use_container_width=True)
            else:
                st.info("No data to visualize")
        
        with tab2:
            # Data table
            viz_data = result.get('visualization_data')
            if viz_data is not None and not viz_data.empty:
                st.dataframe(
                    result['visualization_data'],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv = result['visualization_data'].to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No data available")
        
        with tab3:
            # SQL Query
            st.code(result.get('sql_query', ''), language='sql')
            
            # Execution info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Execution Time", f"{result.get('execution_time', 0):.2f}s")
            with col2:
                st.metric("Rows Returned", result.get('result_count', 0))
            with col3:
                status = "✅ Valid" if result.get('results_valid', False) else "⚠️ Invalid"
                st.metric("Validation", status)
        
        with tab4:
            # Details
            st.subheader("Explanation")
            st.write(result.get('explanation', 'No explanation available'))
            
            st.subheader("Visualization Config")
            st.json(result.get('visualization', {}))
            
            st.subheader("Validation Checks")
            if result.get('validation_checks'):
                for check_name, check_result in result['validation_checks'].get('checks', {}).items():
                    status_icon = "✅" if check_result.get('passed') else "❌"
                    st.write(f"{status_icon} **{check_name.replace('_', ' ').title()}**: {check_result.get('message')}")
    
    elif result.get('status') == 'error':
        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

# Query history sidebar
if st.session_state.query_history:
    with st.sidebar:
        st.markdown("---")
        st.subheader("📜 Recent Queries")
        
        # Show last 5 queries
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