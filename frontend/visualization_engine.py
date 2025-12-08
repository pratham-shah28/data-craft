# backend/visualization_engine.py
"""
Visualization Engine - Creates Plotly charts from data
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


class VisualizationEngine:
    """
    Create interactive Plotly visualizations based on data and config
    """
    
    def create_visualization(self, data: pd.DataFrame, viz_config: dict, query: str = ""):
        """
        Create Plotly figure from data and configuration
        
        Args:
            data: DataFrame with query results
            viz_config: Visualization configuration from model
            query: Original user query (for context)
            
        Returns:
            Plotly figure object
        """
        viz_type = viz_config.get('type', 'table')
        title = viz_config.get('title', 'Query Results')
        
        if viz_type == 'bar_chart':
            return self._create_bar_chart(data, viz_config, title)
        elif viz_type == 'line_chart':
            return self._create_line_chart(data, viz_config, title)
        elif viz_type == 'pie_chart':
            return self._create_pie_chart(data, viz_config, title)
        elif viz_type == 'scatter_plot':
            return self._create_scatter_plot(data, viz_config, title)
        else:
            return self._create_table(data, title)
    
    def _create_bar_chart(self, data: pd.DataFrame, config: dict, title: str):
        """Create bar chart"""
        if len(data) == 0:
            return self._create_empty_figure("No data to display")
        
        # Auto-detect columns if not specified
        x_col = config.get('x_axis') or data.columns[0]
        y_col = config.get('y_axis') or data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=title,
            labels={x_col: config.get('x_label', x_col), y_col: config.get('y_label', y_col)},
            color=y_col,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            template='plotly_white',
            hovermode='x unified',
            showlegend=False
        )
        
        return fig
    
    def _create_line_chart(self, data: pd.DataFrame, config: dict, title: str):
        """Create line chart"""
        if len(data) == 0:
            return self._create_empty_figure("No data to display")
        
        x_col = config.get('x_axis') or data.columns[0]
        y_col = config.get('y_axis') or data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        fig = px.line(
            data,
            x=x_col,
            y=y_col,
            title=title,
            labels={x_col: config.get('x_label', x_col), y_col: config.get('y_label', y_col)},
            markers=True
        )
        
        fig.update_traces(line_color='#1f77b4', line_width=3)
        fig.update_layout(template='plotly_white', hovermode='x unified')
        
        return fig
    
    def _create_pie_chart(self, data: pd.DataFrame, config: dict, title: str):
        """Create pie chart"""
        if len(data) == 0:
            return self._create_empty_figure("No data to display")
        
        labels_col = config.get('labels') or data.columns[0]
        values_col = config.get('values') or data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        fig = px.pie(
            data,
            names=labels_col,
            values=values_col,
            title=title,
            hole=0.3  # Donut chart
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template='plotly_white')
        
        return fig
    
    def _create_scatter_plot(self, data: pd.DataFrame, config: dict, title: str):
        """Create scatter plot"""
        if len(data) == 0:
            return self._create_empty_figure("No data to display")
        
        x_col = config.get('x_axis') or data.columns[0]
        y_col = config.get('y_axis') or data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            title=title,
            labels={x_col: config.get('x_label', x_col), y_col: config.get('y_label', y_col)},
            trendline="ols" if len(data) > 3 else None
        )
        
        fig.update_traces(marker=dict(size=10, opacity=0.6))
        fig.update_layout(template='plotly_white')
        
        return fig
    
    def _create_table(self, data: pd.DataFrame, title: str):
        """Create table visualization"""
        if len(data) == 0:
            return self._create_empty_figure("No data to display")
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(data.columns),
                fill_color='#1f77b4',
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[data[col] for col in data.columns],
                fill_color='#f0f2f6',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(title=title, template='plotly_white')
        
        return fig
    
    def _create_empty_figure(self, message: str):
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            template='plotly_white',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig