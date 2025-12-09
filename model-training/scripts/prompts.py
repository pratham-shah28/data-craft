# scripts/prompts.py
"""
Prompt Engineering for Query & Visualization Model
Contains few-shot examples and system prompts for Gemini
"""

from typing import Dict, List

# ========================================
# FEW-SHOT EXAMPLES
# ========================================

FEW_SHOT_EXAMPLES = """
Example 1:
User Query: "What is the total sales by region?"
Dataset Schema: 
- region (categorical, 4 unique values: East, West, North, South)
- sales (numeric, range: $100-$50000)

Generated SQL:
SELECT 
    region,
    SUM(sales) as total_sales
FROM dataset
GROUP BY region
ORDER BY total_sales DESC;

Visualization Config:
{
    "type": "bar_chart",
    "x_axis": "region",
    "y_axis": "total_sales",
    "title": "Total Sales by Region",
    "x_label": "Region",
    "y_label": "Total Sales ($)"
}

Explanation: Aggregating sales by region to show regional performance comparison.

---

Example 2:
User Query: "Show me the trend of profit over time"
Dataset Schema:
- order_date (datetime, range: 2020-01-01 to 2023-12-31)
- profit (numeric, range: -$5000 to $15000)

Generated SQL:
SELECT 
    DATE_TRUNC(order_date, MONTH) as month,
    AVG(profit) as avg_profit
FROM dataset
GROUP BY month
ORDER BY month;

Visualization Config:
{
    "type": "line_chart",
    "x_axis": "month",
    "y_axis": "avg_profit",
    "title": "Profit Trend Over Time",
    "x_label": "Month",
    "y_label": "Average Profit ($)"
}

Explanation: Time series analysis showing monthly profit trends using date truncation.

---

Example 3:
User Query: "Compare sales across different customer segments"
Dataset Schema:
- segment (categorical, 3 unique values: Consumer, Corporate, Home Office)
- sales (numeric, range: $100-$50000)

Generated SQL:
SELECT 
    segment,
    SUM(sales) as total_sales,
    COUNT(*) as order_count
FROM dataset
GROUP BY segment
ORDER BY total_sales DESC;

Visualization Config:
{
    "type": "pie_chart",
    "labels": "segment",
    "values": "total_sales",
    "title": "Sales Distribution by Customer Segment",
    "show_percentage": true
}

Explanation: Pie chart shows proportional distribution of sales across customer segments.

---

Example 4:
User Query: "What are the top 10 products by profit?"
Dataset Schema:
- product_name (categorical, 1850 unique values)
- profit (numeric, range: -$5000 to $15000)

Generated SQL:
SELECT 
    product_name,
    SUM(profit) as total_profit
FROM dataset
GROUP BY product_name
ORDER BY total_profit DESC
LIMIT 10;

Visualization Config:
{
    "type": "bar_chart",
    "x_axis": "product_name",
    "y_axis": "total_profit",
    "title": "Top 10 Products by Profit",
    "x_label": "Product Name",
    "y_label": "Total Profit ($)",
    "orientation": "horizontal"
}

Explanation: Horizontal bar chart makes product names more readable for top-N analysis.

---

Example 5:
User Query: "Show relationship between discount and profit"
Dataset Schema:
- discount (numeric, range: 0-0.8)
- profit (numeric, range: -$5000 to $15000)

Generated SQL:
SELECT 
    ROUND(discount, 2) as discount_rate,
    AVG(profit) as avg_profit,
    COUNT(*) as transaction_count
FROM dataset
GROUP BY discount_rate
ORDER BY discount_rate;

Visualization Config:
{
    "type": "scatter_plot",
    "x_axis": "discount_rate",
    "y_axis": "avg_profit",
    "size": "transaction_count",
    "title": "Discount vs Profit Analysis",
    "x_label": "Discount Rate",
    "y_label": "Average Profit ($)"
}

Explanation: Scatter plot reveals correlation between discount levels and profitability.

---

Example 6:
User Query: "List all orders with details"
Dataset Schema:
- order_id (categorical, 5009 unique values)
- customer_name (categorical, 793 unique values)
- sales (numeric)
- profit (numeric)

Generated SQL:
SELECT 
    order_id,
    customer_name,
    sales,
    profit
FROM dataset
ORDER BY order_id
LIMIT 100;

Visualization Config:
{
    "type": "table",
    "title": "Order Details",
    "sortable": true,
    "page_size": 25
}

Explanation: Table view appropriate for detailed record listing with pagination.
"""

# ========================================
# SYSTEM PROMPT TEMPLATE
# ========================================

SYSTEM_PROMPT_TEMPLATE = """You are an expert data analyst assistant that generates SQL queries and visualization configurations based on natural language questions.

**CRITICAL RULES:**

**RULE 1 - Data Analysis Questions:**
For questions about the dataset, follow these guidelines:

1. Generate valid BigQuery SQL
2. ALWAYS use the exact table name provided in the Dataset Context below
3. NEVER use placeholder names like 'dataset' or 'table_name'
4. Use appropriate aggregations (SUM, AVG, COUNT, MIN, MAX)
5. Add ORDER BY clauses for better readability
6. Use LIMIT when showing top-N results
7. Choose the most appropriate visualization type
8. **For date operations, ALWAYS use CAST(column AS DATE) instead of PARSE_DATE or PARSE_TIMESTAMP**
9. **For date grouping, use: DATE_TRUNC(CAST(date_column AS DATE), MONTH)**
10. Return response in VALID JSON format only (no markdown, no backticks)

**Date Handling Examples:**
- ✅ CORRECT: `CAST(date_column AS DATE)`
- ✅ CORRECT: `DATE_TRUNC(CAST(date_column AS DATE), MONTH)`
- ✅ CORRECT: `CAST(date_column AS DATE) >= DATE '2024-01-01'`
- ❌ WRONG: `PARSE_DATE('%Y-%m-%d', date_column)`
- ❌ WRONG: `PARSE_TIMESTAMP('%m/%d/%Y', date_column)`

**Dataset Context:**
{dataset_context}

**Available Visualization Types:**
- bar_chart: For comparing categories or showing rankings
- line_chart: For time series or trend analysis
- pie_chart: For showing proportions/distributions
- scatter_plot: For showing relationships between two numeric variables
- table: For detailed record listings
- none: For non-data questions (use with sql_query: null)

**RULE 2 - Personal/Conversational Questions:**
If the user asks a personal question (about you, your preferences, opinions, feelings) or asks something NOT related to the dataset, you MUST respond with this EXACT format:
{{
    "sql_query": null,
    "visualization": {{"type": "none"}},
    "explanation": "I can only answer questions about your dataset. Please ask about the data, such as: 'What are the top 10 records?' or 'Show me trends over time.'"
}}

Examples of questions to reject:
- "What is your favorite food?"
- "How are you?"
- "What do you think about X?"
- Any question not related to analyzing the dataset

**Response Format (JSON only, no markdown):**

For DATA questions:
{{
    "sql_query": "SELECT ... FROM `full_table_name` ...",
    "visualization": {{
        "type": "bar_chart|line_chart|pie_chart|scatter_plot|table",
        "x_axis": "column_name",
        "y_axis": "column_name",
        "title": "Chart Title",
        "x_label": "X Axis Label",
        "y_label": "Y Axis Label"
    }},
    "explanation": "Brief explanation of the analysis"
}}

For PERSONAL/NON-DATA questions:
{{
    "sql_query": null,
    "visualization": {{"type": "none"}},
    "explanation": "I can only answer questions about your dataset. Please ask about the data."
}}

**Few-Shot Examples:**
{few_shot_examples}

Now, generate SQL and visualization for this query:
User Query: "{user_query}"

Remember: If this is a personal/conversational question, return sql_query as null with type "none".
"""

# ========================================
# HELPER FUNCTIONS
# ========================================

def build_prompt(
    user_query: str,
    dataset_context: str,
    few_shot_examples: str = FEW_SHOT_EXAMPLES
) -> str:
    """
    Build complete prompt with context and examples
    
    Args:
        user_query: User's natural language question
        dataset_context: Rich context about the dataset (from metadata_manager)
        few_shot_examples: Few-shot examples (default: FEW_SHOT_EXAMPLES)
    
    Returns:
        Complete prompt string ready for Gemini
        
    Example:
        >>> prompt = build_prompt(
        ...     "What are top products?",
        ...     metadata['llm_context']
        ... )
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        dataset_context=dataset_context,
        few_shot_examples=few_shot_examples,
        user_query=user_query
    )


def validate_response_format(response: Dict) -> bool:
    """
    Validate that LLM response has required fields
    
    Args:
        response: Parsed JSON response from LLM
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["sql_query", "visualization", "explanation"]
    
    if not all(field in response for field in required_fields):
        return False
    
    viz_required = ["type", "title"]
    if not all(field in response["visualization"] for field in viz_required):
        return False
    
    valid_types = ["bar_chart", "line_chart", "pie_chart", "scatter_plot", "table"]
    if response["visualization"]["type"] not in valid_types:
        return False
    
    return True


def get_visualization_requirements(viz_type: str) -> List[str]:
    """
    Get required fields for each visualization type
    
    Args:
        viz_type: Type of visualization
        
    Returns:
        List of required field names
    """
    requirements = {
        "bar_chart": ["x_axis", "y_axis", "title"],
        "line_chart": ["x_axis", "y_axis", "title"],
        "pie_chart": ["labels", "values", "title"],
        "scatter_plot": ["x_axis", "y_axis", "title"],
        "table": ["title"]
    }
    
    return requirements.get(viz_type, ["title"])


# ========================================
# PROMPT TEMPLATES FOR SPECIFIC QUERIES
# ========================================

AGGREGATION_QUERIES = [
    "total sales by region",
    "sum of profit by category",
    "average discount by segment",
    "count orders by ship mode"
]

TIME_SERIES_QUERIES = [
    "sales trend over time",
    "monthly profit analysis",
    "quarterly revenue growth",
    "daily order volume"
]

TOP_N_QUERIES = [
    "top 10 products",
    "best customers",
    "highest profit orders",
    "most popular categories"
]

COMPARISON_QUERIES = [
    "compare segments",
    "region vs region",
    "product performance",
    "category breakdown"
]


def get_query_category(user_query: str) -> str:
    """
    Categorize user query to help with prompt selection
    
    Args:
        user_query: User's natural language question
        
    Returns:
        Category string: 'aggregation', 'time_series', 'top_n', 'comparison', 'other'
    """
    query_lower = user_query.lower()
    
    if any(keyword in query_lower for keyword in ['top', 'best', 'highest', 'lowest']):
        return 'top_n'
    
    if any(keyword in query_lower for keyword in ['trend', 'over time', 'monthly', 'daily', 'yearly']):
        return 'time_series'
    
    if any(keyword in query_lower for keyword in ['compare', 'vs', 'versus', 'comparison']):
        return 'comparison'
    
    if any(keyword in query_lower for keyword in ['total', 'sum', 'average', 'count']):
        return 'aggregation'
    
    return 'other'


# ========================================
# TESTING
# ========================================

if __name__ == "__main__":
    """Test prompt building"""
    
    # Sample dataset context
    sample_context = """
Dataset: orders
Total Records: 51,290
Total Columns: 23

Column Definitions:
- order_id: Order Identifier (categorical, 5009 unique values)
- order_date: Order Date Timestamp (datetime)
- sales: Sales Amount (numeric, range: $0.44-$22638.48)
- profit: Profit Amount (numeric, range: -$6599.98-$8399.98)
- region: Region Location (categorical, 4 unique values)
- segment: Segment Information (categorical, 3 unique values)
- category: Category Details (categorical, 3 unique values)
- product_name: Product Name Details (categorical, 1850 unique values)
"""
    
    # Test queries
    test_queries = [
        "What are the top 10 products by sales?",
        "Show me sales trend over time",
        "Compare profit across regions",
        "What is the average discount by segment?"
    ]
    
    print("\n" + "=" * 60)
    print("TESTING PROMPT ENGINEERING")
    print("=" * 60 + "\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print(f"   Category: {get_query_category(query)}")
        
        prompt = build_prompt(query, sample_context)
        print(f"   Prompt length: {len(prompt)} characters")
        print(f"   Preview: {prompt[:200]}...")
    
    # Test validation
    print("\n\nTesting Response Validation:")
    
    valid_response = {
        "sql_query": "SELECT region, SUM(sales) FROM dataset GROUP BY region",
        "visualization": {
            "type": "bar_chart",
            "x_axis": "region",
            "y_axis": "total_sales",
            "title": "Sales by Region"
        },
        "explanation": "Aggregating sales by region"
    }
    
    invalid_response = {
        "sql_query": "SELECT * FROM dataset",
        "visualization": {
            "type": "invalid_type"
        }
    }
    
    print(f"   Valid response: {validate_response_format(valid_response)}")
    print(f"   Invalid response: {validate_response_format(invalid_response)}")
    
    print("\n" + "=" * 60)
    print("✓ PROMPT ENGINEERING TEST PASSED!")
    print("=" * 60 + "\n")