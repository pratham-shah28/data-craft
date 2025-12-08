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
    
    def generate_description(self, dataset_name: str) -> Dict:
        """
        Generate a natural language description of the dataset using LLM
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Dictionary with description, insights, and metadata
        """
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
            # Configure generation for structured output
            generation_config = GenerationConfig(
                temperature=0.3,  # Lower temperature for more structured output
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
            )
            
            # Generate description using Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            response_text = response.text
            
            # Clean response text - remove markdown code blocks
            response_text = response_text.strip()
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Parse the response (expecting JSON format)
            try:
                description_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON response: {e}")
                self.logger.warning(f"Raw response: {response_text[:200]}")
                
                # If not valid JSON, treat as plain text
                description_data = {
                    'overview': response_text if len(response_text) < 500 else response_text[:500] + "...",
                    'key_insights': ["Data analysis in progress"],
                    'suggested_analyses': ["Further investigation needed"],
                    'data_quality': {
                        'completeness': 'Unknown',
                        'consistency': 'Unknown',
                        'notes': 'Unable to parse full analysis'
                    }
                }
            
            return {
                'status': 'success',
                'dataset_name': dataset_name,
                'description': description_data.get('overview', response_text[:200]),
                'key_insights': description_data.get('key_insights', []),
                'suggested_analyses': description_data.get('suggested_analyses', []),
                'data_quality': description_data.get('data_quality', {}),
                'metadata': {
                    'row_count': metadata['row_count'],
                    'column_count': metadata['column_count'],
                    'size_mb': round(metadata['size_bytes'] / (1024 * 1024), 2),
                    'last_modified': metadata['modified']
                }
            }
            
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
        
        prompt = f"""Generate exactly 5 questions about this dataset.

Dataset: {dataset_name}
Columns: {columns_list}
Row Count: {metadata['row_count']:,}

Requirements:
- Questions should be specific to the actual columns
- Vary complexity: aggregations, comparisons, trends, top N
- Business user language (no SQL terms)
- Each question should be clear and actionable

Return ONLY a JSON array:
["question1", "question2", "question3", "question4", "question5"]

Example:
["What are the top 5 products by revenue?", "How do sales vary by region?", "Show customer distribution by segment", "Which month had highest orders?", "Compare average values across categories"]"""

        try:
            generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=2096,  # Increased from 512
                response_mime_type="application/json"
            )
            
            response = self.model.generate_content(prompt, generation_config=generation_config)
            response_text = response.text.strip()
            
            self.logger.info(f"Raw query response length: {len(response_text)}")
            
            # Clean response
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Extract JSON array
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx+1]
                
                # Parse JSON
                queries = json.loads(json_str)
                
                if isinstance(queries, list) and len(queries) >= 5:
                    # Ensure all queries are strings and clean
                    queries = [str(q).strip() for q in queries if q][:5]
                    
                    # Cache the queries
                    self.save_cached_queries(dataset_name, queries)
                    
                    self.logger.info(f"Successfully generated {len(queries)} queries for {dataset_name}")
                    return queries
            
            raise ValueError("Could not extract valid JSON array")
                
        except Exception as e:
            self.logger.warning(f"Failed to generate sample queries: {str(e)}")
            if 'response_text' in locals():
                self.logger.warning(f"Response text: {response_text[:200]}")
            
            # Return and cache fallback queries
            fallback = self._get_fallback_queries(dataset_name)
            self.save_cached_queries(dataset_name, fallback)
            return fallback
    
    def _get_fallback_queries(self, dataset_name: str) -> list:
        """Get fallback queries for a dataset"""
        return [
            f"Show me a summary of {dataset_name}",
            f"What are the top 10 records in {dataset_name}?",
            f"Give me key insights from {dataset_name}",
            f"What patterns exist in {dataset_name}?",
            f"Analyze trends in {dataset_name}"
        ]
    
    def get_cached_queries(self, dataset_name: str) -> list:
        """
        Retrieve cached sample queries from BigQuery metadata
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            List of cached queries or None
        """
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
        """
        Save sample queries to BigQuery metadata table
        
        Args:
            dataset_name: Name of the dataset
            queries: List of query strings
        """
        try:
            # First check if metadata exists
            check_query = f"""
            SELECT COUNT(*) as count
            FROM `{self.project_id}.{self.dataset_id}.dataset_metadata`
            WHERE dataset_name = @dataset_name
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("dataset_name", "STRING", dataset_name)
                ]
            )
            
            result = self.bq_client.query(check_query, job_config=job_config).result()
            exists = list(result)[0].count > 0
            
            if exists:
                # Update existing record
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
                self.logger.info(f"Cached queries for {dataset_name}")
            else:
                self.logger.warning(f"Metadata record not found for {dataset_name}, skipping cache")
            
        except Exception as e:
            self.logger.warning(f"Could not cache queries: {str(e)}")
            # Don't fail if caching doesn't work
    
    def _create_description_prompt(self, metadata: Dict) -> str:
        """
        Create a prompt for the LLM to generate dataset description
        
        Args:
            metadata: Dataset metadata dictionary
        
        Returns:
            Formatted prompt string
        """
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
        
        # Build stats section separately to avoid backslash in f-string
        stats_section = ""
        if stats_str:
            stats_section = f"Statistical Summary:\n{stats_str}\n\n"
        
        prompt = f"""Analyze this dataset and provide a comprehensive description in JSON format.

Dataset: {metadata['table_name']}
Total Rows: {metadata['row_count']:,}
Total Columns: {metadata['column_count']}
Size: {round(metadata['size_bytes'] / (1024 * 1024), 2)} MB

Columns:
{columns_info}

Sample Data (first 3 rows):
{sample_str}

{stats_section}IMPORTANT: Respond with ONLY a valid JSON object. Do not include any markdown formatting, code blocks, or additional text.

Required JSON structure:
{{
  "overview": "A 2-3 sentence overview of what this dataset contains and its primary purpose",
  "key_insights": [
    "Insight 1 about the data patterns or structure",
    "Insight 2 about data quality or completeness",
    "Insight 3 about interesting characteristics"
  ],
  "suggested_analyses": [
    "Suggested analysis 1 based on the data",
    "Suggested analysis 2 that would be valuable",
    "Suggested analysis 3 for business insights"
  ],
  "data_quality": {{
    "completeness": "Good/Fair/Poor assessment",
    "consistency": "Good/Fair/Poor assessment",
    "notes": "Any quality concerns or observations"
  }}
}}

Be specific and insightful. Focus on what makes this dataset useful and what questions it can answer.

Respond with JSON only:"""

        return prompt
    
    def get_cached_description(self, dataset_name: str) -> Optional[Dict]:
        """
        Check if a description already exists in BigQuery metadata
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Cached description if exists, None otherwise
        """
        # This could be extended to store descriptions in a separate table
        # For now, return None to always generate fresh descriptions
        return None
    
    def save_description(self, dataset_name: str, description: Dict):
        """
        Save generated description for future use
        
        Args:
            dataset_name: Name of the dataset
            description: Description dictionary to save
        """
        # This could be extended to save descriptions to a BigQuery table
        # For now, just log
        self.logger.info(f"Description generated for {dataset_name}")