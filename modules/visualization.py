#!/usr/bin/env python3
"""
SmolVLM-GeoEye Visualization Module
===================================

Advanced visualization engine for geotechnical data.
Creates interactive charts and graphs from extracted data.

Author: SmolVLM-GeoEye Team
Version: 3.1.0
"""

import logging
from typing import Dict, List, Any, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dataclasses import asdict

logger = logging.getLogger(__name__)

class GeotechnicalVisualizationEngine:
    """Advanced visualization engine for geotechnical data"""
    
    def __init__(self):
        self.default_colors = {
            'spt': '#1f77b4',
            'bearing': '#ff7f0e',
            'density': '#2ca02c',
            'moisture': '#d62728',
            'cohesion': '#9467bd',
            'friction': '#8c564b',
            'settlement': '#e377c2',
            'primary': '#636EFA',
            'secondary': '#EF553B',
            'tertiary': '#00CC96',
        }
        
    def create_visualization_from_any_document(self, doc_data: Dict[str, Any]) -> go.Figure:
        """Create visualization based on document type and available data"""
        doc_type = doc_data.get('document_type', 'unknown')
        numerical_data = doc_data.get('numerical_data', {})
        
        if not numerical_data or not any(numerical_data.values()):
            return self._create_no_data_figure()
        
        # Choose visualization based on available data
        if 'spt_values' in numerical_data and numerical_data['spt_values']:
            return self.create_spt_depth_profile(numerical_data)
        elif any(param in numerical_data for param in ['bearing_capacity', 'density', 'moisture_content']):
            return self.create_parameter_distribution(numerical_data)
        else:
            return self.create_multi_parameter_chart(numerical_data)
    
    def create_spt_depth_profile(self, numerical_data: Dict[str, List[Any]]) -> go.Figure:
        """Create SPT N-value vs depth profile"""
        spt_values = numerical_data.get('spt_values', [])
        
        if not spt_values:
            return self._create_no_data_figure()
        
        # Extract depths and values
        depths = []
        values = []
        
        for item in spt_values:
            if hasattr(item, 'depth') and item.depth is not None:
                depths.append(-abs(item.depth))  # Negative for downward
                values.append(item.value)
            else:
                # If no depth, use sequential depths
                depths.append(-len(depths) * 1.5)
                values.append(item.value)
        
        # Create figure
        fig = go.Figure()
        
        # Add SPT profile
        fig.add_trace(go.Scatter(
            x=values,
            y=depths,
            mode='lines+markers',
            name='SPT N-values',
            line=dict(color=self.default_colors['spt'], width=3),
            marker=dict(size=10, symbol='circle')
        ))
        
        # Add soil consistency zones
        fig.add_vrect(x0=0, x1=10, fillcolor="red", opacity=0.1, 
                     annotation_text="Very Loose/Soft", annotation_position="top right")
        fig.add_vrect(x0=10, x1=30, fillcolor="orange", opacity=0.1,
                     annotation_text="Medium Dense", annotation_position="top right")
        fig.add_vrect(x0=30, x1=50, fillcolor="green", opacity=0.1,
                     annotation_text="Dense", annotation_position="top right")
        fig.add_vrect(x0=50, x1=100, fillcolor="darkgreen", opacity=0.1,
                     annotation_text="Very Dense", annotation_position="top right")
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'SPT N-Value Profile with Depth',
                'font': {'size': 24, 'color': '#2c3e50'}
            },
            xaxis_title='SPT N-Value (blows/ft)',
            yaxis_title='Depth (m)',
            xaxis=dict(
                range=[0, max(max(values) * 1.2, 60)],
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                gridcolor='lightgray',
                showgrid=True,
                zeroline=True,
                zerolinecolor='black',
                zerolinewidth=2
            ),
            plot_bgcolor='white',
            hovermode='x unified',
            height=600
        )
        
        return fig
    
    def create_parameter_distribution(self, numerical_data: Dict[str, List[Any]]) -> go.Figure:
        """Create distribution charts for various parameters"""
        # Create subplots
        num_params = sum(1 for v in numerical_data.values() if v)
        rows = (num_params + 1) // 2
        cols = 2 if num_params > 1 else 1
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[param.replace('_', ' ').title() 
                          for param, values in numerical_data.items() if values],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Add traces for each parameter
        plot_idx = 0
        for param_type, values in numerical_data.items():
            if not values:
                continue
            
            plot_idx += 1
            row = (plot_idx - 1) // cols + 1
            col = (plot_idx - 1) % cols + 1
            
            # Extract numerical values
            nums = [v.value for v in values if hasattr(v, 'value')]
            
            if nums:
                # Add histogram
                fig.add_trace(
                    go.Histogram(
                        x=nums,
                        name=param_type.replace('_', ' ').title(),
                        marker_color=self.default_colors.get(param_type.split('_')[0], 
                                                            self.default_colors['primary']),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add mean line
                mean_val = np.mean(nums)
                fig.add_vline(
                    x=mean_val,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {mean_val:.2f}",
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Geotechnical Parameter Distributions',
                'font': {'size': 24, 'color': '#2c3e50'}
            },
            showlegend=False,
            plot_bgcolor='white',
            height=300 * rows
        )
        
        # Update axes
        fig.update_xaxes(title_text="Value", gridcolor='lightgray')
        fig.update_yaxes(title_text="Frequency", gridcolor='lightgray')
        
        return fig
    
    def create_multi_parameter_chart(self, numerical_data: Dict[str, List[Any]]) -> go.Figure:
        """Create multi-parameter comparison chart"""
        # Prepare data for visualization
        param_stats = []
        
        for param_type, values in numerical_data.items():
            if values:
                nums = [v.value for v in values if hasattr(v, 'value')]
                if nums:
                    param_stats.append({
                        'Parameter': param_type.replace('_', ' ').title(),
                        'Min': min(nums),
                        'Mean': np.mean(nums),
                        'Max': max(nums),
                        'Count': len(nums)
                    })
        
        if not param_stats:
            return self._create_no_data_figure()
        
        df = pd.DataFrame(param_stats)
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Parameter Ranges', 'Sample Counts'),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # Add range plot
        for idx, row in df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row['Parameter'], row['Parameter'], row['Parameter']],
                    y=[row['Min'], row['Mean'], row['Max']],
                    mode='lines+markers',
                    name=row['Parameter'],
                    marker=dict(size=[8, 12, 8]),
                    line=dict(width=3)
                ),
                row=1, col=1
            )
        
        # Add count bars
        fig.add_trace(
            go.Bar(
                x=df['Parameter'],
                y=df['Count'],
                name='Sample Count',
                marker_color=self.default_colors['primary'],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Geotechnical Parameters Summary',
                'font': {'size': 24, 'color': '#2c3e50'}
            },
            plot_bgcolor='white',
            height=700,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(tickangle=-45, gridcolor='lightgray')
        fig.update_yaxes(title_text="Value Range", row=1, col=1, gridcolor='lightgray')
        fig.update_yaxes(title_text="Count", row=2, col=1, gridcolor='lightgray')
        
        return fig
    
    def create_correlation_matrix(self, numerical_data: Dict[str, List[Any]]) -> go.Figure:
        """Create correlation matrix for parameters with common depths"""
        # Prepare data frame with aligned depths
        depth_data = {}
        
        for param_type, values in numerical_data.items():
            for v in values:
                if hasattr(v, 'depth') and v.depth is not None:
                    depth = round(v.depth, 1)  # Round to nearest 0.1m
                    if depth not in depth_data:
                        depth_data[depth] = {}
                    depth_data[depth][param_type] = v.value
        
        if not depth_data:
            return self._create_no_data_figure()
        
        # Create DataFrame
        df = pd.DataFrame.from_dict(depth_data, orient='index')
        df = df.dropna(thresh=2)  # Keep rows with at least 2 values
        
        if df.shape[1] < 2:
            return self._create_no_data_figure()
        
        # Calculate correlation
        corr = df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Parameter Correlation Matrix',
                'font': {'size': 24, 'color': '#2c3e50'}
            },
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'},
            width=700,
            height=700,
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_3d_soil_profile(self, numerical_data: Dict[str, List[Any]], 
                              x_coords: Optional[List[float]] = None,
                              y_coords: Optional[List[float]] = None) -> go.Figure:
        """Create 3D soil profile visualization"""
        spt_values = numerical_data.get('spt_values', [])
        
        if not spt_values:
            return self._create_no_data_figure()
        
        # Generate grid if coordinates not provided
        if not x_coords or not y_coords:
            n_points = len(spt_values)
            grid_size = int(np.sqrt(n_points)) + 1
            x_coords = np.repeat(np.arange(grid_size), grid_size)[:n_points]
            y_coords = np.tile(np.arange(grid_size), grid_size)[:n_points]
        
        # Extract data
        x = []
        y = []
        z = []
        values = []
        
        for idx, item in enumerate(spt_values):
            if idx < len(x_coords) and idx < len(y_coords):
                x.append(x_coords[idx])
                y.append(y_coords[idx])
                z.append(-abs(item.depth) if hasattr(item, 'depth') and item.depth else -idx)
                values.append(item.value)
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=8,
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="SPT N-Value")
            ),
            text=[f'SPT: {v}' for v in values],
            hovertemplate='X: %{x}<br>Y: %{y}<br>Depth: %{z}m<br>%{text}<extra></extra>'
        )])
        
        # Update layout
        fig.update_layout(
            title={
                'text': '3D Soil Profile Visualization',
                'font': {'size': 24, 'color': '#2c3e50'}
            },
            scene=dict(
                xaxis_title='X Coordinate (m)',
                yaxis_title='Y Coordinate (m)',
                zaxis_title='Depth (m)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    
    def create_time_series_analysis(self, historical_data: List[Dict[str, Any]]) -> go.Figure:
        """Create time series analysis of parameters"""
        if not historical_data:
            return self._create_no_data_figure()
        
        # Prepare time series data
        timestamps = []
        param_series = {}
        
        for entry in historical_data:
            timestamp = entry.get('timestamp')
            if timestamp:
                timestamps.append(pd.to_datetime(timestamp))
                
                numerical_data = entry.get('numerical_data', {})
                for param_type, values in numerical_data.items():
                    if values:
                        if param_type not in param_series:
                            param_series[param_type] = []
                        avg_value = np.mean([v.value for v in values if hasattr(v, 'value')])
                        param_series[param_type].append(avg_value)
        
        if not timestamps or not param_series:
            return self._create_no_data_figure()
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each parameter
        for param_type, values in param_series.items():
            fig.add_trace(go.Scatter(
                x=timestamps[:len(values)],
                y=values,
                mode='lines+markers',
                name=param_type.replace('_', ' ').title(),
                line=dict(width=2)
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Parameter Trends Over Time',
                'font': {'size': 24, 'color': '#2c3e50'}
            },
            xaxis_title='Date',
            yaxis_title='Average Value',
            plot_bgcolor='white',
            hovermode='x unified',
            height=500
        )
        
        # Update axes
        fig.update_xaxes(gridcolor='lightgray')
        fig.update_yaxes(gridcolor='lightgray')
        
        return fig
    
    def _create_no_data_figure(self) -> go.Figure:
        """Create a figure indicating no data available"""
        fig = go.Figure()
        
        fig.add_annotation(
            text="No numerical data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        
        fig.update_layout(
            xaxis_visible=False,
            yaxis_visible=False,
            plot_bgcolor='white',
            height=400
        )
        
        return fig
    
    def export_figure(self, fig: go.Figure, filename: str, format: str = 'png'):
        """Export figure to file"""
        supported_formats = ['png', 'jpg', 'svg', 'pdf', 'html']
        
        if format not in supported_formats:
            logger.warning(f"Unsupported format {format}. Using PNG.")
            format = 'png'
        
        try:
            if format == 'html':
                fig.write_html(filename)
            else:
                fig.write_image(filename, format=format)
            logger.info(f"Figure exported to {filename}")
        except Exception as e:
            logger.error(f"Failed to export figure: {e}")
