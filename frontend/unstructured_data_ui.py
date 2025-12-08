"""
Unstructured Data UI Component - Streamlit interface for PDF/Image processing
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import os
from datetime import datetime


from unstructured_data_handler import UnstructuredDataHandler


def render_unstructured_tab(config: dict, handler, viz_engine, dataset_manager):
    """
    Render the Unstructured Data tab
    
    Args:
        config: Configuration dictionary
        handler: QueryHandler instance (for querying)
        viz_engine: VisualizationEngine instance
        dataset_manager: DatasetManager instance
    """
    
    # Initialize unstructured handler
    if 'unstructured_handler' not in st.session_state:
        st.session_state.unstructured_handler = UnstructuredDataHandler(
            project_id=config['project_id'],
            dataset_id=config['dataset_id'],
            region=config['region']
        )
    
    unstructured_handler = st.session_state.unstructured_handler
    
    st.markdown("## 📄 Unstructured Data Processing")
    st.markdown("Upload and extract data from PDFs, invoices, insurance documents, and more.")
    
    # Two column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Documents")
        
        # Document type selection
        doc_type = st.text_input(
            "Document Type",
            value="invoices",
            help="e.g., invoices, insurance, receipts, contracts",
            key="unstructured_doc_type"
        )
        
        # Examples directory (optional)
        st.markdown("**Optional: Examples Directory**")
        examples_dir = st.text_input(
            "Path to example documents",
            value="",
            help="Directory containing example PDFs and their JSONs for few-shot learning",
            key="unstructured_examples_dir"
        )
        
        # File uploader - multiple files
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="unstructured_file_uploader"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Show file list
            with st.expander("📋 Uploaded Files", expanded=False):
                for file in uploaded_files:
                    st.text(f"• {file.name} ({file.size / 1024:.1f} KB)")
        
        # Process button
        if st.button("🚀 Process Documents", type="primary", disabled=not uploaded_files):
            process_documents(
                uploaded_files,
                doc_type,
                examples_dir if examples_dir else None,
                unstructured_handler
            )
    
    with col2:
        st.subheader("📊 Processed Data")
        
        # List existing unstructured tables
        tables = unstructured_handler.list_unstructured_tables()
        
        if tables:
            st.markdown(f"**Found {len(tables)} unstructured dataset(s)**")
            
            for table in tables:
                with st.expander(f"📄 {table['name'].title()}", expanded=False):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Rows", f"{table['rows']:,}")
                        st.metric("Columns", table['columns'])
                    with col_b:
                        st.metric("Size", f"{table['size_mb']} MB")
                        st.caption(f"Updated: {table['updated'][:10] if table['updated'] else 'N/A'}")
                    
                    # Query button for this table
                    if st.button(f"🔍 Query {table['name']}", key=f"query_{table['name']}"):
                        st.session_state.selected_unstructured_table = table['name']
                        st.rerun()
        else:
            st.info("No unstructured datasets found. Upload documents to get started.")
    
    # Query section (if table selected)
    if 'selected_unstructured_table' in st.session_state:
        render_query_section(
            st.session_state.selected_unstructured_table,
            handler,
            viz_engine,
            config
        )


def process_documents(uploaded_files, doc_type, examples_dir, unstructured_handler):
    """
    Process uploaded PDF documents
    """
    with st.spinner(f"🤖 Processing {len(uploaded_files)} document(s)..."):
        try:
            # Save uploaded files temporarily
            temp_paths = []
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    temp_path = os.path.join(temp_dir, file.name)
                    with open(temp_path, 'wb') as f:
                        f.write(file.read())
                    temp_paths.append(temp_path)
                
                # Process all files
                extracted_data = unstructured_handler.process_multiple_files(
                    temp_paths,
                    doc_type,
                    examples_dir
                )
            
            # Display extraction results
            st.success(f"✅ Successfully extracted data from {len(extracted_data)} document(s)")
            
            # Show preview
            with st.expander("📄 Extracted Data Preview", expanded=True):
                for i, data in enumerate(extracted_data[:3]):  # Show first 3
                    st.markdown(f"**Document {i+1}:**")
                    st.json(data)
                
                if len(extracted_data) > 3:
                    st.info(f"... and {len(extracted_data) - 3} more documents")
            
            # Store in BigQuery
            with st.spinner("💾 Storing data in BigQuery..."):
                table_id = unstructured_handler.store_in_bigquery(
                    extracted_data,
                    doc_type
                )
            
            st.success(f"✅ Data stored in BigQuery: `{table_id}`")
            st.balloons()
            
            # Refresh page to show new table
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            st.exception(e)


def render_query_section(table_name, handler, viz_engine, config):
    """
    Render query interface for unstructured data table
    """
    st.markdown("---")
    st.subheader(f"🔍 Query: {table_name.title()}")
    
    # Update handler to use this table
    handler.table_name = f"{table_name}_processed"
    
    # Query input
    user_query = st.text_input(
        "Ask a question about your data:",
        placeholder=f"e.g., What is the total amount from all {table_name}?",
        key=f"query_input_{table_name}"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    
    with col_btn1:
        query_button = st.button("🚀 Run Query", type="primary", key=f"run_query_{table_name}")
    
    with col_btn2:
        if st.button("⬅️ Back", key=f"back_{table_name}"):
            del st.session_state.selected_unstructured_table
            st.rerun()
    
    # Process query
    if query_button and user_query:
        with st.spinner("🤖 Processing query..."):
            try:
                result = handler.process_query(user_query, table_name)
                
                if result.get('status') == 'success':
                    st.success("✅ Query executed successfully!")
                    
                    # Answer
                    st.markdown("### 💬 Answer")
                    st.markdown(f"**{result.get('natural_language_answer', 'No answer')}**")
                    
                    # Tabs for results
                    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📋 Data", "🔍 SQL"])
                    
                    with tab1:
                        viz_data = result.get('visualization_data')
                        if viz_data is not None and not viz_data.empty:
                            try:
                                fig = viz_engine.create_visualization(
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
                                file_name=f"{table_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    with tab3:
                        st.code(result.get('sql_query', ''), language='sql')
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Execution Time", f"{result.get('execution_time', 0):.2f}s")
                        with col2:
                            st.metric("Rows", result.get('result_count', 0))
                        with col3:
                            status = "✅ Valid" if result.get('results_valid') else "⚠️ Invalid"
                            st.metric("Status", status)
                
                else:
                    st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")