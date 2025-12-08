# backend/visualization_engine.py
"""
Visualization Engine - Creates Plotly charts from data
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


class VisualizationEngine:
    """
    Create interactive Plotly visualizations based on data and config
    """
    
    # Define a professional color palette
    COLORS = [
        '#1f77b4', '#9467bd', '#2ca02c', '#ff7f0e', '#d62728',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
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
        """Create bar chart with improved visibility and proper text display"""
        if len(data) == 0:
            return self._create_empty_figure("No data to display")
        
        # Auto-detect columns if not specified
        x_col = config.get('x_axis') or data.columns[0]
        y_col = config.get('y_axis') or data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        # Create discrete color palette for better visibility
        num_bars = len(data)
        colors = [self.COLORS[i % len(self.COLORS)] for i in range(num_bars)]
        
        # Calculate dynamic margin for text labels
        max_value = data[y_col].max()
        top_margin = 120 if max_value > 1000 else 100
        
        fig = go.Figure(data=[
            go.Bar(
                x=data[x_col],
                y=data[y_col],
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=1)
                ),
                text=data[y_col],
                texttemplate='%{text:,.0f}',  # Format with commas, no decimals
                textposition='outside',
                textfont=dict(size=12, color='#333', family='Arial'),
                hovertemplate='<b>%{x}</b><br>%{y:,.2f}<extra></extra>',
                cliponaxis=False  # Don't clip text labels
            )
        ])
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='#333')),
            template='plotly_white',
            hovermode='x unified',
            showlegend=False,
            xaxis=dict(
                title=config.get('x_label', x_col),
                tickangle=-45 if num_bars > 5 else 0
            ),
            yaxis=dict(
                title=config.get('y_label', y_col),
                range=[0, max_value * 1.15]  # Add 15% padding for text labels
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='white',
            margin=dict(t=top_margin, b=100, l=80, r=40),
            height=500,
            uniformtext=dict(mode='hide', minsize=8)  # Ensure text visibility
        )
        
        return fig
    
    def _create_line_chart(self, data: pd.DataFrame, config: dict, title: str):
        """Create line chart"""
        if len(data) == 0:
            return self._create_empty_figure("No data to display")
        
        x_col = config.get('x_axis') or data.columns[0]
        y_col = config.get('y_axis') or data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        fig = go.Figure(data=[
            go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8, color='#1f77b4', line=dict(color='white', width=2)),
                hovertemplate='<b>%{x}</b><br>%{y:,.2f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='#333')),
            template='plotly_white',
            hovermode='x unified',
            xaxis=dict(title=config.get('x_label', x_col)),
            yaxis=dict(title=config.get('y_label', y_col)),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='white',
            margin=dict(t=80, b=80, l=80, r=40),
            height=500
        )
        
        return fig
    
    def _create_pie_chart(self, data: pd.DataFrame, config: dict, title: str):
        """Create pie chart with distinct colors"""
        if len(data) == 0:
            return self._create_empty_figure("No data to display")
        
        labels_col = config.get('labels') or data.columns[0]
        values_col = config.get('values') or data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=data[labels_col],
                values=data[values_col],
                hole=0.3,
                marker=dict(
                    colors=self.COLORS,
                    line=dict(color='white', width=2)
                ),
                textposition='inside',
                textinfo='percent+label',
                textfont=dict(size=12),
                hovertemplate='<b>%{label}</b><br>%{value:,.2f}<br>%{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='#333')),
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            ),
            margin=dict(t=80, b=40, l=40, r=150),
            height=500
        )
        
        return fig
    
    def _create_scatter_plot(self, data: pd.DataFrame, config: dict, title: str):
        """Create scatter plot without statsmodels dependency"""
        if len(data) == 0:
            return self._create_empty_figure("No data to display")
        
        x_col = config.get('x_axis') or data.columns[0]
        y_col = config.get('y_axis') or data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        fig = go.Figure(data=[
            go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='markers',
                marker=dict(
                    size=10,
                    color='#1f77b4',
                    opacity=0.6,
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<extra></extra>'
            )
        ])
        
        # Add simple linear trendline if data has enough points
        if len(data) > 3:
            try:
                # Simple linear regression without statsmodels
                x_numeric = pd.to_numeric(data[x_col], errors='coerce').dropna()
                y_numeric = pd.to_numeric(data[y_col], errors='coerce').dropna()
                
                if len(x_numeric) > 3 and len(y_numeric) > 3:
                    # Calculate trendline using numpy
                    coeffs = np.polyfit(x_numeric, y_numeric, 1)
                    trendline = np.poly1d(coeffs)
                    
                    x_trend = np.linspace(x_numeric.min(), x_numeric.max(), 100)
                    y_trend = trendline(x_trend)
                    
                    fig.add_trace(go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        name='Trend',
                        line=dict(color='red', width=2, dash='dash'),
                        hoverinfo='skip'
                    ))
            except:
                pass  # Skip trendline if calculation fails
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='#333')),
            template='plotly_white',
            xaxis=dict(title=config.get('x_label', x_col)),
            yaxis=dict(title=config.get('y_label', y_col)),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='white',
            showlegend=True,
            margin=dict(t=80, b=80, l=80, r=40),
            height=500
        )
        
        return fig
    
    def _create_table(self, data: pd.DataFrame, title: str):
        """Create table visualization with improved styling"""
        if len(data) == 0:
            return self._create_empty_figure("No data to display")
        
        # Format numeric columns
        formatted_data = data.copy()
        for col in formatted_data.columns:
            if pd.api.types.is_numeric_dtype(formatted_data[col]):
                formatted_data[col] = formatted_data[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[f"<b>{col}</b>" for col in data.columns],
                fill_color='#1f77b4',
                font=dict(color='white', size=13),
                align='left',
                height=40
            ),
            cells=dict(
                values=[formatted_data[col] for col in formatted_data.columns],
                fill_color=[['#f9f9f9', '#ffffff'] * len(data)],
                align='left',
                font=dict(size=12),
                height=35
            )
        )])
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='#333')),
            template='plotly_white',
            margin=dict(t=80, b=40, l=40, r=40),
            height=min(600, 100 + len(data) * 35)
        )
        
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
            font=dict(size=20, color='#666')
        )
        fig.update_layout(
            template='plotly_white',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig