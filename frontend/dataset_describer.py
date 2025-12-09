"""
Dataset Describer - Generate natural language descriptions of datasets using LLM
"""

import pandas as pd
from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import json
from typing import Dict, Optional
import logging

class DatasetDescriber:
    def __init__(self, project_id: str, dataset_id: str, region: str = "us-central1", model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize the Dataset Describer
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            region: GCP region for Vertex AI
            model_name: Gemini model to use
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.region = region
        self.model_name = model_name
        
        # Initialize BigQuery client
        self.bq_client = bigquery.Client(project=project_id)
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel(model_name)
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure metadata table has description columns
        self._ensure_description_columns()
    
    def _ensure_description_columns(self):
        """Ensure metadata table has columns for storing descriptions"""
        try:
            # Check if columns exist
            table_id = f"{self.project_id}.{self.dataset_id}.dataset_metadata"
            table = self.bq_client.get_table(table_id)
            
            existing_fields = {field.name for field in table.schema}
            
            # Add missing columns if needed
            new_fields = []
            
            if 'description_overview' not in existing_fields:
                new_fields.append(bigquery.SchemaField('description_overview', 'STRING'))
            if 'key_insights' not in existing_fields:
                new_fields.append(bigquery.SchemaField('key_insights', 'STRING'))  # JSON array
            if 'suggested_analyses' not in existing_fields:
                new_fields.append(bigquery.SchemaField('suggested_analyses', 'STRING'))  # JSON array
            if 'data_quality' not in existing_fields:
                new_fields.append(bigquery.SchemaField('data_quality', 'STRING'))  # JSON object
            
            if new_fields:
                # Update schema
                new_schema = list(table.schema) + new_fields
                table.schema = new_schema
                table = self.bq_client.update_table(table, ['schema'])
                self.logger.info("✓ Updated metadata table schema for descriptions")
        
        except Exception as e:
            self.logger.warning(f"Could not update metadata schema: {str(e)}")
    
    def get_dataset_metadata(self, dataset_name: str) -> Dict:
        """
        Fetch dataset metadata from BigQuery
        
        Args:
            dataset_name: Name of the dataset (table name without _processed suffix)
        
        Returns:
            Dictionary containing metadata
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{dataset_name}_processed"
        
        try:
            # Get table metadata
            table = self.bq_client.get_table(table_id)
            
            # Get column information with types
            columns_info = []
            for field in table.schema:
                columns_info.append({
                    'name': field.name,
                    'type': field.field_type,
                    'mode': field.mode
                })
            
            # Get row count
            query = f"SELECT COUNT(*) as total FROM `{table_id}`"
            result = self.bq_client.query(query).result()
            row_count = list(result)[0].total
            
            # Get sample data (first 5 rows)
            sample_query = f"SELECT * FROM `{table_id}` LIMIT 5"
            sample_result = self.bq_client.query(sample_query).result()
            sample_df = sample_result.to_dataframe()
            
            # Get column statistics for numeric columns
            numeric_columns = [col['name'] for col in columns_info if col['type'] in ['INTEGER', 'FLOAT', 'NUMERIC', 'BIGNUMERIC']]
            
            stats = {}
            if numeric_columns:
                stats_queries = []
                for col in numeric_columns[:5]:  # Limit to first 5 numeric columns
                    stats_queries.append(f"""
                        MIN(`{col}`) as {col}_min,
                        MAX(`{col}`) as {col}_max,
                        AVG(`{col}`) as {col}_avg
                    """)
                
                stats_query = f"SELECT {', '.join(stats_queries)} FROM `{table_id}`"
                stats_result = self.bq_client.query(stats_query).result()
                stats_row = list(stats_result)[0]
                stats = dict(stats_row.items())
            
            metadata = {
                'table_name': dataset_name,
                'row_count': row_count,
                'column_count': len(columns_info),
                'columns': columns_info,
                'sample_data': sample_df.to_dict('records'),
                'statistics': stats,
                'created': table.created.isoformat() if table.created else None,
                'modified': table.modified.isoformat() if table.modified else None,
                'size_bytes': table.num_bytes
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error fetching metadata for {dataset_name}: {str(e)}")
            return None
    
    def get_cached_description(self, dataset_name: str) -> Optional[Dict]:
        """
        Check if a description already exists in BigQuery metadata
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Cached description if exists, None otherwise
        """
        try:
            query = f"""
            SELECT 
                description_overview,
                key_insights,
                suggested_analyses,
                data_quality
            FROM `{self.project_id}.{self.dataset_id}.dataset_metadata`
            WHERE dataset_name = @dataset_name
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("dataset_name", "STRING", dataset_name)
                ]
            )
            
            result = self.bq_client.query(query, job_config=job_config).result()
            
            for row in result:
                # Check if description exists
                if row.description_overview:
                    self.logger.info(f"✓ Found cached description for {dataset_name}")
                    
                    return {
                        'overview': row.description_overview,
                        'key_insights': json.loads(row.key_insights) if row.key_insights else [],
                        'suggested_analyses': json.loads(row.suggested_analyses) if row.suggested_analyses else [],
                        'data_quality': json.loads(row.data_quality) if row.data_quality else {}
                    }
            
            self.logger.info(f"No cached description found for {dataset_name}")
            return None
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve cached description: {str(e)}")
            return None
    
    def save_description(self, dataset_name: str, description_data: Dict):
        """
        Save generated description to BigQuery metadata
        
        Args:
            dataset_name: Name of the dataset
            description_data: Description dictionary with overview, key_insights, etc.
        """
        try:
            update_query = f"""
            UPDATE `{self.project_id}.{self.dataset_id}.dataset_metadata`
            SET 
                description_overview = @overview,
                key_insights = @key_insights,
                suggested_analyses = @suggested_analyses,
                data_quality = @data_quality,
                updated_at = CURRENT_TIMESTAMP()
            WHERE dataset_name = @dataset_name
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("dataset_name", "STRING", dataset_name),
                    bigquery.ScalarQueryParameter("overview", "STRING", description_data.get('overview', '')),
                    bigquery.ScalarQueryParameter("key_insights", "STRING", json.dumps(description_data.get('key_insights', []))),
                    bigquery.ScalarQueryParameter("suggested_analyses", "STRING", json.dumps(description_data.get('suggested_analyses', []))),
                    bigquery.ScalarQueryParameter("data_quality", "STRING", json.dumps(description_data.get('data_quality', {})))
                ]
            )
            
            self.bq_client.query(update_query, job_config=job_config).result()
            self.logger.info(f"✓ Saved description for {dataset_name}")
            
        except Exception as e:
            self.logger.error(f"Could not save description: {str(e)}")
    
    def generate_description(self, dataset_name: str, force_regenerate: bool = False) -> Dict:
        """
        Generate a natural language description of the dataset using LLM
        
        Args:
            dataset_name: Name of the dataset
            force_regenerate: If True, bypass cache and regenerate
        
        Returns:
            Dictionary with description, insights, and metadata
        """
        # ✅ CHECK CACHE FIRST (unless force regenerate)
        if not force_regenerate:
            cached = self.get_cached_description(dataset_name)
            if cached:
                self.logger.info(f"Using cached description for {dataset_name}")
                
                # Get basic metadata for stats
                metadata = self.get_dataset_metadata(dataset_name)
                
                return {
                    'status': 'success',
                    'dataset_name': dataset_name,
                    'description': cached['overview'],
                    'key_insights': cached['key_insights'],
                    'suggested_analyses': cached['suggested_analyses'],
                    'data_quality': cached['data_quality'],
                    'metadata': {
                        'row_count': metadata['row_count'] if metadata else 0,
                        'column_count': metadata['column_count'] if metadata else 0,
                        'size_mb': round(metadata['size_bytes'] / (1024 * 1024), 2) if metadata else 0,
                        'last_modified': metadata['modified'] if metadata else None
                    },
                    'from_cache': True
                }
        
        # ✅ GENERATE NEW DESCRIPTION
        self.logger.info(f"Generating new description for {dataset_name}")
        
        # Get metadata
        metadata = self.get_dataset_metadata(dataset_name)
        
        if not metadata:
            return {
                'status': 'error',
                'error': f'Could not fetch metadata for dataset: {dataset_name}'
            }
        
        # Prepare prompt for LLM
        prompt = self._create_description_prompt(metadata)
        
        try:
            # ✅ Use response_mime_type for guaranteed JSON output
            generation_config = GenerationConfig(
                temperature=0.3,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
                response_mime_type="application/json"  # Forces JSON output
            )
            
            # Generate description using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            response_text = response.text.strip()
            
            self.logger.info(f"Raw response length: {len(response_text)} chars")
            
            # ✅ Clean response text - remove markdown if present
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # ✅ Parse the response
            try:
                description_data = json.loads(response_text)
                self.logger.info("✓ Successfully parsed JSON response")
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON: {e}")
                self.logger.error(f"Raw response: {response_text[:500]}")
                
                # ✅ Fallback: Create structured response from text
                description_data = {
                    'overview': f"Dataset with {metadata['row_count']:,} rows and {metadata['column_count']} columns",
                    'key_insights': [
                        f"Contains {metadata['row_count']:,} records",
                        f"Has {metadata['column_count']} different fields",
                        "Ready for analysis"
                    ],
                    'suggested_analyses': [
                        f"Explore the distribution of values",
                        f"Analyze patterns across columns",
                        f"Investigate relationships in the data"
                    ],
                    'data_quality': {
                        'completeness': 'Good',
                        'consistency': 'Good',
                        'notes': 'Metadata analysis in progress'
                    }
                }
            
            # ✅ SAVE TO CACHE for future use
            self.save_description(dataset_name, description_data)
            
            result = {
                'status': 'success',
                'dataset_name': dataset_name,
                'description': description_data.get('overview', 'No description available'),
                'key_insights': description_data.get('key_insights', []),
                'suggested_analyses': description_data.get('suggested_analyses', []),
                'data_quality': description_data.get('data_quality', {}),
                'metadata': {
                    'row_count': metadata['row_count'],
                    'column_count': metadata['column_count'],
                    'size_mb': round(metadata['size_bytes'] / (1024 * 1024), 2),
                    'last_modified': metadata['modified']
                },
                'from_cache': False
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating description: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def generate_sample_queries(self, dataset_name: str, metadata: Dict = None) -> list:
        """
        Generate sample queries for a dataset using LLM with caching
        
        Args:
            dataset_name: Name of the dataset
            metadata: Optional pre-fetched metadata (to avoid duplicate calls)
        
        Returns:
            List of 5 sample query strings
        """
        # Check cache first
        cached_queries = self.get_cached_queries(dataset_name)
        if cached_queries:
            self.logger.info(f"Using cached queries for {dataset_name}")
            return cached_queries
        
        # Get metadata if not provided
        if not metadata:
            metadata = self.get_dataset_metadata(dataset_name)
        
        if not metadata:
            # Fallback queries
            fallback = self._get_fallback_queries(dataset_name)
            self.save_cached_queries(dataset_name, fallback)
            return fallback
        
        # Create prompt for query generation
        columns_list = ", ".join([col['name'] for col in metadata['columns'][:10]])
        
        prompt = f"""Generate exactly 5 natural language questions about this dataset.

Dataset: {dataset_name}
Columns: {columns_list}
Row Count: {metadata['row_count']:,}

Requirements:
- Questions should be specific to the actual columns
- Vary complexity: aggregations, comparisons, trends, top N
- Business user language (no SQL terms)
- Each question should be clear and actionable
- Make them practical and useful

Return ONLY a JSON array with 5 questions:
["question1", "question2", "question3", "question4", "question5"]"""

        try:
            generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=1024,
                response_mime_type="application/json"
            )
            
            response = self.model.generate_content(prompt, generation_config=generation_config)
            response_text = response.text.strip()
            
            # Clean response
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            queries = json.loads(response_text)
            
            if isinstance(queries, list) and len(queries) >= 5:
                # Ensure all queries are strings and clean
                queries = [str(q).strip() for q in queries if q][:5]
                
                # Cache the queries
                self.save_cached_queries(dataset_name, queries)
                
                self.logger.info(f"✓ Generated {len(queries)} queries for {dataset_name}")
                return queries
            
            raise ValueError("Invalid query format")
                
        except Exception as e:
            self.logger.warning(f"Failed to generate sample queries: {str(e)}")
            
            # Return and cache fallback queries
            fallback = self._get_fallback_queries(dataset_name)
            self.save_cached_queries(dataset_name, fallback)
            return fallback
    
    def _get_fallback_queries(self, dataset_name: str) -> list:
        """Get fallback queries for a dataset"""
        return [
            f"Show me the first 10 rows from {dataset_name}",
            f"What is the total count of records in {dataset_name}?",
            f"Give me a summary of {dataset_name}",
            f"What are the most common values in {dataset_name}?",
            f"Show me key statistics from {dataset_name}"
        ]
    
    def get_cached_queries(self, dataset_name: str) -> list:
        """Retrieve cached sample queries from BigQuery metadata"""
        try:
            query = f"""
            SELECT sample_queries
            FROM `{self.project_id}.{self.dataset_id}.dataset_metadata`
            WHERE dataset_name = @dataset_name
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("dataset_name", "STRING", dataset_name)
                ]
            )
            
            result = self.bq_client.query(query, job_config=job_config).result()
            
            for row in result:
                if row.sample_queries:
                    queries = json.loads(row.sample_queries)
                    if isinstance(queries, list) and len(queries) > 0:
                        return queries
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Could not retrieve cached queries: {str(e)}")
            return None
    
    def save_cached_queries(self, dataset_name: str, queries: list):
        """Save sample queries to BigQuery metadata table"""
        try:
            update_query = f"""
            UPDATE `{self.project_id}.{self.dataset_id}.dataset_metadata`
            SET sample_queries = @queries,
                updated_at = CURRENT_TIMESTAMP()
            WHERE dataset_name = @dataset_name
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("dataset_name", "STRING", dataset_name),
                    bigquery.ScalarQueryParameter("queries", "STRING", json.dumps(queries))
                ]
            )
            
            self.bq_client.query(update_query, job_config=job_config).result()
            self.logger.info(f"✓ Cached queries for {dataset_name}")
            
        except Exception as e:
            self.logger.warning(f"Could not cache queries: {str(e)}")
    
    def _create_description_prompt(self, metadata: Dict) -> str:
        """Create a prompt for the LLM to generate dataset description"""
        # Format columns with types
        columns_info = "\n".join([
            f"  - {col['name']} ({col['type']})" 
            for col in metadata['columns']
        ])
        
        # Format sample data
        sample_df = pd.DataFrame(metadata['sample_data'])
        sample_str = sample_df.head(3).to_string(index=False)
        
        # Format statistics
        stats_str = ""
        if metadata['statistics']:
            stats_str = "\n".join([
                f"  - {key}: {value}" 
                for key, value in list(metadata['statistics'].items())[:10]
            ])
        
        stats_section = ""
        if stats_str:
            stats_section = f"Statistical Summary:\n{stats_str}\n\n"
        
        prompt = f"""Analyze this dataset and provide a comprehensive description.

Dataset: {metadata['table_name']}
Total Rows: {metadata['row_count']:,}
Total Columns: {metadata['column_count']}
Size: {round(metadata['size_bytes'] / (1024 * 1024), 2)} MB

Columns:
{columns_info}

Sample Data (first 3 rows):
{sample_str}

{stats_section}Return a JSON object with this EXACT structure:
{{
  "overview": "A 2-3 sentence overview of what this dataset contains",
  "key_insights": [
    "First insight about the data",
    "Second insight about patterns",
    "Third insight about characteristics"
  ],
  "suggested_analyses": [
    "First suggested analysis",
    "Second suggested analysis",
    "Third suggested analysis"
  ],
  "data_quality": {{
    "completeness": "Good/Fair/Poor",
    "consistency": "Good/Fair/Poor",
    "notes": "Brief quality assessment"
  }}
}}

Be specific and concise. Focus on actionable insights."""

        return prompt