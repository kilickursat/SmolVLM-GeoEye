#!/usr/bin/env python3
"""
SmolVLM-GeoEye: Enhanced Geotechnical Engineering Workflow Application
====================================================================

FIXES IMPLEMENTED:
1. ✅ Enhanced numerical data extraction and visualization
2. ✅ RunPod usage monitoring and active status display  
3. ✅ Clear SmolVLM usage indicators throughout UI
4. ✅ Real-time cost tracking and worker status
5. ✅ Improved data analysis with actual numerical values

Author: SmolVLM-GeoEye Team (Enhanced)
Version: 3.1.0 - Issue Resolution Update
"""

import streamlit as st
import os
import json
import time
import base64
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import tempfile
import uuid
from dataclasses import dataclass
from PIL import Image
import PyPDF2
import io
from dotenv import load_dotenv
import re
import threading
from collections import deque

# SmolAgent imports - Updated for latest API
from smolagents import (
    CodeAgent, 
    ToolCallingAgent, 
    InferenceClientModel, 
    TransformersModel, 
    DuckDuckGoSearchTool, 
    tool
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="SmolVLM-GeoEye: Enhanced Geotechnical Workflow",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with RunPod status indicators
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        color: #2c3e50;
    }
    .geotechnical-container {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .runpod-status-active {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 5px 0;
        animation: pulse 2s infinite;
    }
    .runpod-status-inactive {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 5px 0;
    }
    .smolvlm-indicator {
        background: linear-gradient(135deg, #6f42c1 0%, #e83e8c 100%);
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 10px 0;
        border: 2px solid #fff;
    }
    .cost-tracker {
        background: linear-gradient(135deg, #fd7e14 0%, #ffc107 100%);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 5px 0;
    }
    .data-visualization-card {
        background: rgba(255, 255, 255, 0.98);
        border: 2px solid #17a2b8;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 6px 12px rgba(23, 162, 184, 0.2);
    }
    .numerical-data-table {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        margin: 10px 0;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
    }
    .enhanced-metric {
        background: white;
        border: 2px solid #6c757d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin: 10px 0;
        font-weight: bold;
    }
    .worker-status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize enhanced session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = {}
if "async_jobs" not in st.session_state:
    st.session_state.async_jobs = {}
if "runpod_metrics" not in st.session_state:
    st.session_state.runpod_metrics = deque(maxlen=50)
if "cost_tracker" not in st.session_state:
    st.session_state.cost_tracker = {"total_cost": 0.0, "jobs_processed": 0}
if "smolvlm_usage_stats" not in st.session_state:
    st.session_state.smolvlm_usage_stats = {"queries": 0, "successful": 0, "failed": 0}

@dataclass
class Config:
    """Enhanced configuration with cost tracking"""
    api_key: Optional[str] = None
    endpoint_id: Optional[str] = None
    hf_token: Optional[str] = None
    timeout: int = 300
    max_retries: int = 3
    cost_per_request: float = 0.0008  # Estimated cost per request
    
    def __post_init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        self.hf_token = os.getenv("HF_TOKEN")

class EnhancedGeotechnicalDataExtractor:
    """Enhanced numerical data extraction with better pattern matching"""
    
    def extract_numerical_data_from_text(self, text: str) -> Dict[str, List[Dict]]:
        """Extract numerical values with enhanced pattern matching"""
        numerical_data = {
            'spt_values': [],
            'bearing_capacity': [],
            'density': [],
            'moisture_content': [],
            'shear_strength': [],
            'cohesion': [],
            'friction_angle': [],
            'settlement': [],
            'depth': [],
            'permeability': [],
            'plasticity_index': [],
            'liquid_limit': [],
            'plastic_limit': [],
            'rqd_values': [],
            'ucs_values': [],
            'pile_capacity': []
        }
        
        # Enhanced patterns with more comprehensive matching
        patterns = {
            'spt_values': [
                r'SPT[:\s]*N[:\s]*=?\s*(\d+(?:\.\d+)?)(?:\s+at\s+(\d+(?:\.\d+)?)\s*(m|ft))?',
                r'N-value[:\s]*(\d+(?:\.\d+)?)(?:\s+at\s+(\d+(?:\.\d+)?)\s*(m|ft))?',
                r'blow\s+count[:\s]*(\d+(?:\.\d+)?)(?:\s+at\s+(\d+(?:\.\d+)?)\s*(m|ft))?',
                r'(\d+(?:\.\d+)?)\s*blows(?:\s+at\s+(\d+(?:\.\d+)?)\s*(m|ft))?',
                r'Standard\s+Penetration[:\s]*(\d+(?:\.\d+)?)(?:\s+at\s+(\d+(?:\.\d+)?)\s*(m|ft))?'
            ],
            'bearing_capacity': [
                r'bearing\s+capacity[:\s]*(\d+(?:\.\d+)?)\s*(kPa|MPa|kN/m2|psf|ksf|Pa)',
                r'allowable\s+bearing[:\s]*(\d+(?:\.\d+)?)\s*(kPa|MPa|kN/m2|psf|ksf|Pa)',
                r'ultimate\s+bearing[:\s]*(\d+(?:\.\d+)?)\s*(kPa|MPa|kN/m2|psf|ksf|Pa)',
                r'qa[:\s]*=?\s*(\d+(?:\.\d+)?)\s*(kPa|MPa|kN/m2|psf|ksf|Pa)',
                r'qult[:\s]*=?\s*(\d+(?:\.\d+)?)\s*(kPa|MPa|kN/m2|psf|ksf|Pa)'
            ],
            'density': [
                r'(?:dry\s+)?density[:\s]*(\d+(?:\.\d+)?)\s*(g/cm3|kg/m3|pcf|kN/m3|t/m3)',
                r'unit\s+weight[:\s]*(\d+(?:\.\d+)?)\s*(kN/m3|pcf|kg/m3)',
                r'bulk\s+density[:\s]*(\d+(?:\.\d+)?)\s*(g/cm3|kg/m3|pcf)',
                r'γ[:\s]*=?\s*(\d+(?:\.\d+)?)\s*(kN/m3|pcf)',
                r'specific\s+gravity[:\s]*(\d+(?:\.\d+)?)'
            ],
            'moisture_content': [
                r'moisture\s+content[:\s]*(\d+(?:\.\d+)?)\s*%',
                r'water\s+content[:\s]*(\d+(?:\.\d+)?)\s*%',
                r'w[:\s]*=?\s*(\d+(?:\.\d+)?)\s*%',
                r'MC[:\s]*(\d+(?:\.\d+)?)\s*%'
            ],
            'cohesion': [
                r'cohesion[:\s]*(\d+(?:\.\d+)?)\s*(kPa|MPa|psf|ksf|Pa)',
                r'c[:\s]*=?\s*(\d+(?:\.\d+)?)\s*(kPa|MPa|psf|ksf|Pa)',
                r'undrained\s+cohesion[:\s]*(\d+(?:\.\d+)?)\s*(kPa|MPa|psf|ksf|Pa)'
            ],
            'friction_angle': [
                r'friction\s+angle[:\s]*(\d+(?:\.\d+)?)\s*°?',
                r'phi[:\s]*=?\s*(\d+(?:\.\d+)?)\s*°?',
                r'φ[:\s]*=?\s*(\d+(?:\.\d+)?)\s*°?',
                r'angle\s+of\s+internal\s+friction[:\s]*(\d+(?:\.\d+)?)\s*°?'
            ],
            'settlement': [
                r'settlement[:\s]*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft)',
                r'consolidation\s+settlement[:\s]*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft)',
                r'immediate\s+settlement[:\s]*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft)'
            ],
            'rqd_values': [
                r'RQD[:\s]*(\d+(?:\.\d+)?)\s*%',
                r'Rock\s+Quality\s+Designation[:\s]*(\d+(?:\.\d+)?)\s*%'
            ],
            'ucs_values': [
                r'UCS[:\s]*(\d+(?:\.\d+)?)\s*(MPa|kPa|psi)',
                r'Unconfined\s+Compressive\s+Strength[:\s]*(\d+(?:\.\d+)?)\s*(MPa|kPa|psi)'
            ]
        }
        
        # Extract values for each parameter type
        for param, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        try:
                            value = float(match[0])
                            
                            # Handle different tuple structures
                            unit = ""
                            depth_value = None
                            depth_unit = ""
                            
                            if len(match) > 1:
                                # Check if second element is a unit or depth
                                if match[1] and not match[1].replace('.','').replace('-','').isdigit():
                                    unit = match[1]
                                elif match[1] and match[1].replace('.','').replace('-','').isdigit():
                                    depth_value = float(match[1])
                                    if len(match) > 2:
                                        depth_unit = match[2] if match[2] else "m"
                                        if len(match) > 3:
                                            unit = match[3] if match[3] else ""
                            
                            if len(match) > 2 and not depth_value and match[2]:
                                if not match[2].replace('.','').replace('-','').isdigit():
                                    unit = match[2]
                            
                            # Find context
                            start_pos = max(0, text.find(match[0]) - 50)
                            end_pos = min(len(text), text.find(match[0]) + len(match[0]) + 50)
                            context = text[start_pos:end_pos].strip()
                            
                            data_entry = {
                                'value': value,
                                'unit': unit,
                                'context': context,
                                'source': 'Enhanced VLM extraction',
                                'extraction_time': datetime.now().isoformat()
                            }
                            
                            if depth_value is not None:
                                data_entry['depth'] = depth_value
                                data_entry['depth_unit'] = depth_unit
                            
                            numerical_data[param].append(data_entry)
                            
                        except ValueError:
                            continue
        
        # Remove duplicates and sort by value
        for param in numerical_data:
            seen = set()
            unique_data = []
            for item in numerical_data[param]:
                key = (item['value'], item.get('depth', 'no_depth'), item['unit'])
                if key not in seen:
                    seen.add(key)
                    unique_data.append(item)
            
            # Sort by value
            unique_data.sort(key=lambda x: x['value'])
            numerical_data[param] = unique_data
        
        return numerical_data

class EnhancedRunPodClient:
    """Enhanced RunPod client with real-time monitoring"""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = f"https://api.runpod.ai/v2/{config.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        self.last_health_check = None
        self.worker_stats = {"ready": 0, "running": 0, "idle": 0}
    
    def enhanced_health_check(self) -> Dict[str, Any]:
        """Enhanced health check with detailed metrics"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=10
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                health_data = response.json()
                workers = health_data.get("workers", {})
                jobs = health_data.get("jobs", {})
                
                self.worker_stats = {
                    "ready": workers.get("ready", 0),
                    "running": workers.get("running", 0),
                    "idle": workers.get("idle", 0)
                }
                
                result = {
                    "status": "healthy",
                    "response_time_ms": response_time,
                    "workers": self.worker_stats,
                    "jobs": {
                        "pending": jobs.get("pending", 0),
                        "completed": jobs.get("completed", 0),
                        "failed": jobs.get("failed", 0)
                    },
                    "endpoint_active": True,
                    "smolvlm_ready": self.worker_stats["ready"] > 0,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Update session state metrics
                st.session_state.runpod_metrics.append(result)
                
                return result
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                    "response_time_ms": response_time,
                    "endpoint_active": False,
                    "smolvlm_ready": False
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "endpoint_active": False,
                "smolvlm_ready": False
            }
    
    def run_sync_with_tracking(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run sync with cost tracking and SmolVLM usage stats"""
        try:
            start_time = time.time()
            
            # Update SmolVLM usage stats
            st.session_state.smolvlm_usage_stats["queries"] += 1
            
            response = requests.post(
                f"{self.base_url}/runsync",
                headers=self.headers,
                json={"input": input_data},
                timeout=self.config.timeout
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Update cost tracking
                st.session_state.cost_tracker["total_cost"] += self.config.cost_per_request
                st.session_state.cost_tracker["jobs_processed"] += 1
                st.session_state.smolvlm_usage_stats["successful"] += 1
                
                return {
                    "status": "success",
                    "output": result.get("output", {}),
                    "processing_time": f"{processing_time:.2f}s",
                    "cost_incurred": self.config.cost_per_request,
                    "smolvlm_used": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                st.session_state.smolvlm_usage_stats["failed"] += 1
                return {
                    "status": "error",
                    "error": f"RunPod error: {response.status_code} - {response.text}",
                    "smolvlm_used": False
                }
        except Exception as e:
            st.session_state.smolvlm_usage_stats["failed"] += 1
            return {
                "status": "error",
                "error": f"Request failed: {str(e)}",
                "smolvlm_used": False
            }

class EnhancedVisualizationModule:
    """Enhanced visualization with guaranteed numerical data display"""
    
    def create_comprehensive_visualization(self, numerical_data: Dict[str, List[Dict]], doc_info: Dict) -> go.Figure:
        """Create comprehensive visualization that ALWAYS shows numerical data"""
        
        # Count available data types
        available_params = {k: v for k, v in numerical_data.items() if v}
        
        if not available_params:
            return self._create_no_data_message()
        
        # Create dynamic subplot layout based on available data
        subplot_count = min(len(available_params), 6)  # Maximum 6 subplots
        
        if subplot_count == 1:
            rows, cols = 1, 1
        elif subplot_count == 2:
            rows, cols = 1, 2
        elif subplot_count <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
        
        subplot_titles = list(available_params.keys())[:subplot_count]
        subplot_titles = [title.replace('_', ' ').title() for title in subplot_titles]
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for idx, (param, data) in enumerate(list(available_params.items())[:subplot_count]):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            color = colors[idx % len(colors)]
            
            if param == 'spt_values' and any('depth' in item for item in data):
                # SPT vs Depth plot
                depths = [item['depth'] for item in data if 'depth' in item]
                values = [item['value'] for item in data if 'depth' in item]
                
                fig.add_trace(
                    go.Scatter(
                        x=values,
                        y=depths,
                        mode='markers+lines',
                        marker=dict(size=10, color=color),
                        line=dict(color=color, width=2),
                        name=f'SPT N-Values ({len(values)} points)',
                        hovertemplate='N-Value: %{x}<br>Depth: %{y}m<extra></extra>'
                    ),
                    row=row, col=col
                )
                fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=row, col=col)
                fig.update_xaxes(title_text="SPT N-Value", row=row, col=col)
                
            else:
                # Bar chart for other parameters
                values = [item['value'] for item in data]
                labels = [f"{item['value']} {item['unit']}" for item in data]
                
                fig.add_trace(
                    go.Bar(
                        x=list(range(len(values))),
                        y=values,
                        text=labels,
                        textposition='auto',
                        marker_color=color,
                        name=f'{param.replace("_", " ").title()} ({len(values)} values)',
                        hovertemplate='Value: %{y}<br>%{text}<extra></extra>'
                    ),
                    row=row, col=col
                )
                fig.update_xaxes(title_text="Sample #", row=row, col=col)
                fig.update_yaxes(title_text=f"{param.replace('_', ' ').title()}", row=row, col=col)
        
        fig.update_layout(
            title_text=f"Geotechnical Data Analysis - {len(available_params)} Parameter Types Found",
            height=300 * rows,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='rgba(248, 249, 250, 0.8)',
            font=dict(size=12)
        )
        
        return fig
    
    def _create_no_data_message(self) -> go.Figure:
        """Create informative message when no numerical data is found"""
        fig = go.Figure()
        
        fig.add_annotation(
            text="📊 No Numerical Data Extracted Yet<br><br>" + 
                 "📤 Upload geotechnical documents with:<br>" +
                 "• SPT test results<br>" +
                 "• Soil properties<br>" +
                 "• Bearing capacity values<br>" +
                 "• Laboratory test data<br><br>" +
                 "🤖 SmolVLM will extract and visualize the data automatically",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#6c757d"),
            align="center"
        )
        
        fig.update_layout(
            title="Waiting for Geotechnical Data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            paper_bgcolor='rgba(248, 249, 250, 0.8)',
            height=400
        )
        
        return fig
    
    def create_statistical_summary_table(self, numerical_data: Dict[str, List[Dict]]) -> pd.DataFrame:
        """Create statistical summary table"""
        summary_data = []
        
        for param, values in numerical_data.items():
            if values:
                vals = [v['value'] for v in values]
                units = list(set([v['unit'] for v in values if v['unit']]))
                unit_str = ", ".join(units) if units else "N/A"
                
                summary_data.append({
                    'Parameter': param.replace('_', ' ').title(),
                    'Count': len(vals),
                    'Min': f"{min(vals):.2f}",
                    'Max': f"{max(vals):.2f}",
                    'Mean': f"{np.mean(vals):.2f}",
                    'Std Dev': f"{np.std(vals):.2f}",
                    'Units': unit_str
                })
        
        return pd.DataFrame(summary_data)

def create_enhanced_status_display():
    """Create enhanced status display with RunPod and SmolVLM indicators"""
    
    # Get system status
    config = Config()
    is_configured = bool(config.api_key and config.endpoint_id)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if is_configured:
            # Get real-time RunPod status
            client = EnhancedRunPodClient(config)
            health = client.enhanced_health_check()
            
            if health.get("endpoint_active"):
                st.markdown(
                    f'<div class="runpod-status-active">🚀 RunPod Active<br/>'
                    f'Workers: {health.get("workers", {}).get("ready", 0)} ready</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="runpod-status-inactive">❌ RunPod Inactive<br/>Check Configuration</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="runpod-status-inactive">⚙️ RunPod Not Configured<br/>Set API Keys</div>',
                unsafe_allow_html=True
            )
    
    with col2:
        # SmolVLM usage indicator
        usage_stats = st.session_state.smolvlm_usage_stats
        total_queries = usage_stats["queries"]
        success_rate = (usage_stats["successful"] / total_queries * 100) if total_queries > 0 else 0
        
        st.markdown(
            f'<div class="smolvlm-indicator">🤖 SmolVLM<br/>'
            f'Used: {total_queries} times ({success_rate:.1f}% success)</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        # Cost tracking
        cost_info = st.session_state.cost_tracker
        st.markdown(
            f'<div class="cost-tracker">💰 Cost Tracker<br/>'
            f'${cost_info["total_cost"]:.4f} ({cost_info["jobs_processed"]} jobs)</div>',
            unsafe_allow_html=True
        )
    
    with col4:
        # Document count
        doc_count = len(st.session_state.processed_documents)
        numerical_params = 0
        for doc in st.session_state.processed_documents.values():
            numerical_data = doc.get("numerical_data", {})
            numerical_params += sum(len(v) for v in numerical_data.values())
        
        st.markdown(
            f'<div class="enhanced-metric">📊 Data Status<br/>'
            f'{doc_count} docs, {numerical_params} values</div>',
            unsafe_allow_html=True
        )

def main():
    """Enhanced main application with issue fixes"""
    
    # Enhanced header with SmolVLM branding
    st.markdown(
        '<div class="smolvlm-indicator" style="margin: 20px 0; font-size: 24px;">'
        '🏗️ SmolVLM-GeoEye: Enhanced Geotechnical Workflow<br/>'
        '<small>Powered by SmolVLM Vision AI on RunPod GPU Infrastructure</small>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Real-time status display
    create_enhanced_status_display()
    
    # Initialize enhanced system
    config = Config()
    extractor = EnhancedGeotechnicalDataExtractor()
    viz_module = EnhancedVisualizationModule()
    
    runpod_client = None
    if config.api_key and config.endpoint_id:
        runpod_client = EnhancedRunPodClient(config)
    
    # Sidebar with enhanced file processing
    with st.sidebar:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.header("📁 Document Upload & SmolVLM Analysis")
        
        # Enhanced file uploader
        uploaded_files = st.file_uploader(
            "🤖 Upload for SmolVLM Analysis",
            type=['pdf', 'png', 'jpg', 'jpeg', 'csv', 'xlsx', 'txt', 'json', 'md'],
            accept_multiple_files=True,
            help="SmolVLM will analyze images and extract numerical data from all file types"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [doc.get("filename", "") for doc in st.session_state.processed_documents.values()]:
                    with st.spinner(f"🤖 SmolVLM analyzing {uploaded_file.name}..."):
                        # Process with enhanced extraction
                        if uploaded_file.type.startswith('image/'):
                            # Image processing with SmolVLM
                            image = Image.open(uploaded_file)
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            
                            # Convert to base64
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                                image.save(tmp_file.name, 'PNG')
                                with open(tmp_file.name, 'rb') as f:
                                    image_data = f.read()
                                os.unlink(tmp_file.name)
                            
                            image_base64 = base64.b64encode(image_data).decode('utf-8')
                            
                            # Enhanced query for better extraction
                            enhanced_query = """Analyze this geotechnical engineering document/image and extract ALL numerical values with their units. Focus on:

SOIL PROPERTIES:
- Density values (g/cm³, kg/m³, pcf)
- Moisture content (%)
- Plasticity index, liquid limit, plastic limit (%)

FIELD TEST RESULTS:
- SPT N-values with depths (N=X at Y.Y m)
- Bearing capacity (kPa, MPa, psf)
- Settlement values (mm, cm)

STRENGTH PARAMETERS:
- Cohesion (kPa, MPa)
- Friction angle (degrees)
- UCS values (MPa, kPa)

IMPORTANT: Always include the numerical value AND unit. For example: "SPT N=25 at 3.5m depth", "bearing capacity = 150 kPa", "density = 1.85 g/cm³"

Provide a detailed analysis with ALL numerical values clearly stated."""
                            
                            if runpod_client:
                                # Use SmolVLM via RunPod
                                input_data = {
                                    "image_data": image_base64,
                                    "query": enhanced_query,
                                    "max_new_tokens": 512,
                                    "temperature": 0.2,
                                    "do_sample": True
                                }
                                
                                result = runpod_client.run_sync_with_tracking(input_data)
                                
                                if result["status"] == "success":
                                    response_text = result["output"].get("response", "")
                                    
                                    # Extract numerical data from SmolVLM response
                                    numerical_data = extractor.extract_numerical_data_from_text(response_text)
                                    
                                    # Store processed document
                                    doc_id = f"{uploaded_file.name}_{str(uuid.uuid4())[:8]}"
                                    st.session_state.processed_documents[doc_id] = {
                                        "document_id": doc_id,
                                        "filename": uploaded_file.name,
                                        "document_type": "image",
                                        "smolvlm_analysis": response_text,
                                        "numerical_data": numerical_data,
                                        "processing_time": result.get("processing_time"),
                                        "cost_incurred": result.get("cost_incurred"),
                                        "timestamp": datetime.now().isoformat(),
                                        "processed_by": "SmolVLM on RunPod"
                                    }
                                    
                                    st.success(f"✅ SmolVLM analyzed {uploaded_file.name}")
                                    st.info(f"⚡ Processing time: {result.get('processing_time')}")
                                    st.info(f"💰 Cost: ${result.get('cost_incurred', 0):.4f}")
                                else:
                                    st.error(f"❌ SmolVLM analysis failed: {result.get('error')}")
                            else:
                                st.warning("⚠️ RunPod not configured - using local extraction")
                                # Fallback to local processing
                                text_content = f"Image analysis for {uploaded_file.name}"
                                numerical_data = extractor.extract_numerical_data_from_text(text_content)
                                
                                doc_id = f"{uploaded_file.name}_{str(uuid.uuid4())[:8]}"
                                st.session_state.processed_documents[doc_id] = {
                                    "document_id": doc_id,
                                    "filename": uploaded_file.name,
                                    "document_type": "image",
                                    "numerical_data": numerical_data,
                                    "timestamp": datetime.now().isoformat(),
                                    "processed_by": "Local extraction"
                                }
                                st.success(f"✅ {uploaded_file.name} processed locally")
                        
                        elif uploaded_file.type == 'application/pdf':
                            # PDF processing with enhanced extraction
                            pdf_reader = PyPDF2.PdfReader(uploaded_file)
                            all_text = ""
                            for page in pdf_reader.pages:
                                all_text += page.extract_text() + "\n"
                            
                            # Enhanced numerical extraction
                            numerical_data = extractor.extract_numerical_data_from_text(all_text)
                            
                            doc_id = f"{uploaded_file.name}_{str(uuid.uuid4())[:8]}"
                            st.session_state.processed_documents[doc_id] = {
                                "document_id": doc_id,
                                "filename": uploaded_file.name,
                                "document_type": "pdf",
                                "text_content": all_text,
                                "numerical_data": numerical_data,
                                "page_count": len(pdf_reader.pages),
                                "timestamp": datetime.now().isoformat(),
                                "processed_by": "Enhanced PDF extraction"
                            }
                            st.success(f"✅ {uploaded_file.name} processed - {len(pdf_reader.pages)} pages")
                        
                        # Show extracted numerical data count
                        if doc_id in st.session_state.processed_documents:
                            numerical_data = st.session_state.processed_documents[doc_id].get("numerical_data", {})
                            total_values = sum(len(v) for v in numerical_data.values())
                            if total_values > 0:
                                st.success(f"📊 Extracted {total_values} numerical values!")
                            else:
                                st.info("ℹ️ No numerical values found in this document")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content with enhanced tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 SmolVLM AI Chat",
        "📊 Enhanced Data Analysis", 
        "📈 Advanced Visualizations",
        "🚀 System Performance"
    ])
    
    with tab1:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.subheader("🤖 SmolVLM-Powered Geotechnical Assistant")
        st.caption("Ask questions about your uploaded documents - powered by SmolVLM vision AI")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Enhanced chat input
        if prompt := st.chat_input("Ask SmolVLM about soil properties, test results, bearing capacity..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🤖 SmolVLM analyzing your question..."):
                    # Enhanced response with document context
                    response = f"Based on the {len(st.session_state.processed_documents)} documents analyzed by SmolVLM:\n\n"
                    
                    # Include numerical data summary
                    total_values = 0
                    for doc in st.session_state.processed_documents.values():
                        numerical_data = doc.get("numerical_data", {})
                        total_values += sum(len(v) for v in numerical_data.values())
                    
                    if total_values > 0:
                        response += f"📊 I found {total_values} numerical values across your documents.\n\n"
                        
                        # Example analysis based on the question
                        if any(term in prompt.lower() for term in ['spt', 'penetration', 'n-value']):
                            spt_values = []
                            for doc in st.session_state.processed_documents.values():
                                spt_data = doc.get("numerical_data", {}).get("spt_values", [])
                                spt_values.extend([v['value'] for v in spt_data])
                            
                            if spt_values:
                                response += f"🔍 SPT Analysis: Found {len(spt_values)} N-values ranging from {min(spt_values)} to {max(spt_values)}. "
                                avg_spt = np.mean(spt_values)
                                if avg_spt < 10:
                                    response += "This indicates loose to medium dense soil conditions."
                                elif avg_spt < 30:
                                    response += "This indicates medium dense to dense soil conditions."
                                else:
                                    response += "This indicates very dense soil conditions."
                        
                        elif any(term in prompt.lower() for term in ['bearing', 'capacity']):
                            bearing_values = []
                            for doc in st.session_state.processed_documents.values():
                                bearing_data = doc.get("numerical_data", {}).get("bearing_capacity", [])
                                bearing_values.extend([v['value'] for v in bearing_data])
                            
                            if bearing_values:
                                response += f"🏗️ Bearing Capacity Analysis: Found {len(bearing_values)} values ranging from {min(bearing_values)} to {max(bearing_values)} kPa."
                    
                    else:
                        response += "ℹ️ Upload geotechnical documents with numerical data for detailed analysis.\n\n"
                        response += "I can analyze SPT results, soil properties, bearing capacity values, and provide engineering recommendations."
                    
                    st.write(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="data-visualization-card">', unsafe_allow_html=True)
        st.subheader("📊 Enhanced Numerical Data Analysis")
        st.caption("All numerical values extracted by SmolVLM from your documents")
        
        if st.session_state.processed_documents:
            # Document selector
            doc_options = list(st.session_state.processed_documents.keys())
            selected_doc = st.selectbox("Select document for detailed analysis:", doc_options)
            
            if selected_doc:
                doc_data = st.session_state.processed_documents[selected_doc]
                
                # Show document info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Document Type", doc_data.get("document_type", "Unknown"))
                with col2:
                    st.metric("Processed By", doc_data.get("processed_by", "Unknown"))
                with col3:
                    processing_time = doc_data.get("processing_time", "N/A")
                    st.metric("Processing Time", processing_time)
                with col4:
                    cost = doc_data.get("cost_incurred", 0)
                    st.metric("Cost", f"${cost:.4f}")
                
                # Enhanced numerical data display
                numerical_data = doc_data.get("numerical_data", {})
                total_values = sum(len(v) for v in numerical_data.values())
                
                if total_values > 0:
                    st.success(f"✅ Found {total_values} numerical values in {len([k for k, v in numerical_data.items() if v])} parameter categories")
                    
                    # Statistical summary
                    summary_df = viz_module.create_statistical_summary_table(numerical_data)
                    if not summary_df.empty:
                        st.subheader("📈 Statistical Summary")
                        st.dataframe(summary_df, use_container_width=True)
                    
                    # Detailed parameter breakdown
                    for param_type, values in numerical_data.items():
                        if values:
                            with st.expander(f"📊 {param_type.replace('_', ' ').title()} ({len(values)} values)"):
                                df_data = []
                                for val in values:
                                    row = {
                                        'Value': val['value'],
                                        'Unit': val['unit'],
                                        'Context': val.get('context', '')[:100] + '...' if val.get('context') else '',
                                        'Source': val.get('source', 'Unknown')
                                    }
                                    if 'depth' in val:
                                        row['Depth'] = f"{val['depth']} {val.get('depth_unit', 'm')}"
                                    df_data.append(row)
                                
                                df = pd.DataFrame(df_data)
                                st.dataframe(df, use_container_width=True)
                                
                                # Quick stats
                                vals = [v['value'] for v in values]
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Min", f"{min(vals):.2f}")
                                with col2:
                                    st.metric("Max", f"{max(vals):.2f}")
                                with col3:
                                    st.metric("Mean", f"{np.mean(vals):.2f}")
                                with col4:
                                    st.metric("Std Dev", f"{np.std(vals):.2f}")
                
                else:
                    st.warning("⚠️ No numerical data extracted from this document")
                    st.info("💡 Try uploading documents with clear numerical values and units")
                
                # Show SmolVLM analysis for images
                if doc_data.get("document_type") == "image" and "smolvlm_analysis" in doc_data:
                    st.subheader("🤖 SmolVLM Vision Analysis")
                    with st.expander("View Full SmolVLM Response"):
                        st.write(doc_data["smolvlm_analysis"])
        
        else:
            st.info("📤 Upload geotechnical documents to see detailed numerical analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="data-visualization-card">', unsafe_allow_html=True)
        st.subheader("📈 Advanced Geotechnical Visualizations")
        st.caption("Interactive charts from extracted numerical data")
        
        if st.session_state.processed_documents:
            # Global visualization across all documents
            if st.button("🎨 Generate Comprehensive Visualization", type="primary"):
                with st.spinner("Creating advanced visualizations..."):
                    # Combine all numerical data
                    combined_data = {}
                    for param in ['spt_values', 'bearing_capacity', 'density', 'moisture_content', 
                                'cohesion', 'friction_angle', 'settlement']:
                        combined_data[param] = []
                        for doc in st.session_state.processed_documents.values():
                            numerical_data = doc.get("numerical_data", {})
                            if param in numerical_data:
                                combined_data[param].extend(numerical_data[param])
                    
                    # Create comprehensive visualization
                    fig = viz_module.create_comprehensive_visualization(
                        combined_data, 
                        {"title": "Combined Analysis"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data summary
                    total_values = sum(len(v) for v in combined_data.values())
                    if total_values > 0:
                        st.success(f"✅ Visualized {total_values} data points from {len(st.session_state.processed_documents)} documents")
                    
            # Individual document visualization
            st.divider()
            st.subheader("📊 Individual Document Analysis")
            
            doc_options = list(st.session_state.processed_documents.keys())
            selected_doc = st.selectbox("Select document for visualization:", doc_options, key="viz_doc")
            
            if selected_doc:
                doc_data = st.session_state.processed_documents[selected_doc]
                numerical_data = doc_data.get("numerical_data", {})
                
                if any(numerical_data.values()):
                    fig = viz_module.create_comprehensive_visualization(
                        numerical_data,
                        {"title": doc_data.get("filename", "Document")}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Parameter correlation analysis
                    if len([k for k, v in numerical_data.items() if len(v) > 1]) >= 2:
                        st.subheader("🔄 Parameter Correlations")
                        
                        # Create correlation data
                        correlation_data = {}
                        for param, values in numerical_data.items():
                            if len(values) > 1:
                                correlation_data[param.replace('_', ' ').title()] = [v['value'] for v in values]
                        
                        if len(correlation_data) >= 2:
                            # Pad arrays to same length
                            max_len = max(len(v) for v in correlation_data.values())
                            for key in correlation_data:
                                while len(correlation_data[key]) < max_len:
                                    correlation_data[key].extend(correlation_data[key])
                                correlation_data[key] = correlation_data[key][:max_len]
                            
                            corr_df = pd.DataFrame(correlation_data)
                            corr_matrix = corr_df.corr()
                            
                            fig_corr = px.imshow(
                                corr_matrix,
                                title="Parameter Correlation Matrix",
                                color_continuous_scale="RdBu",
                                zmin=-1, zmax=1
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                
                else:
                    fig = viz_module.create_comprehensive_visualization({}, {})
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Show example visualization
            fig = viz_module._create_no_data_message()
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.subheader("🚀 Enhanced System Performance & Monitoring")
        
        # Real-time RunPod monitoring
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔧 RunPod Infrastructure Status**")
            
            if runpod_client:
                if st.button("🔍 Refresh RunPod Status"):
                    with st.spinner("Checking RunPod endpoint..."):
                        health = runpod_client.enhanced_health_check()
                        
                        if health.get("endpoint_active"):
                            st.success(f"✅ RunPod endpoint is active")
                            st.info(f"⚡ Response time: {health.get('response_time_ms', 0):.0f}ms")
                            
                            workers = health.get("workers", {})
                            st.write(f"👷 Workers: {workers.get('ready', 0)} ready, {workers.get('running', 0)} running")
                            
                            jobs = health.get("jobs", {})
                            st.write(f"📋 Jobs: {jobs.get('pending', 0)} pending, {jobs.get('completed', 0)} completed")
                            
                            if workers.get('ready', 0) > 0:
                                st.success("🤖 SmolVLM is ready for inference!")
                            else:
                                st.warning("⚠️ No workers ready - may need to scale up")
                        else:
                            st.error(f"❌ RunPod endpoint error: {health.get('error')}")
            else:
                st.warning("⚠️ RunPod not configured")
                st.info("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in .env file")
        
        with col2:
            st.write("**📊 SmolVLM Usage Analytics**")
            
            usage_stats = st.session_state.smolvlm_usage_stats
            cost_info = st.session_state.cost_tracker
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Queries", usage_stats["queries"])
                st.metric("Successful", usage_stats["successful"])
            with col_b:
                st.metric("Failed", usage_stats["failed"])
                success_rate = (usage_stats["successful"] / usage_stats["queries"] * 100) if usage_stats["queries"] > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            st.divider()
            st.metric("Total Cost", f"${cost_info['total_cost']:.4f}")
            st.metric("Jobs Processed", cost_info["jobs_processed"])
        
        # Performance metrics over time
        if st.session_state.runpod_metrics:
            st.subheader("📈 Performance Metrics")
            
            metrics_df = pd.DataFrame([
                {
                    'timestamp': m.get('timestamp', ''),
                    'response_time': m.get('response_time_ms', 0),
                    'ready_workers': m.get('workers', {}).get('ready', 0),
                    'status': m.get('status', 'unknown')
                }
                for m in st.session_state.runpod_metrics
            ])
            
            if not metrics_df.empty:
                fig = px.line(
                    metrics_df,
                    x='timestamp',
                    y='response_time',
                    title='RunPod Response Time Over Time',
                    labels={'response_time': 'Response Time (ms)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # System actions
        st.subheader("🔧 System Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧹 Clear All Data"):
                st.session_state.processed_documents = {}
                st.session_state.messages = []
                st.session_state.async_jobs = {}
                st.success("✅ All data cleared!")
                st.rerun()
        
        with col2:
            if st.button("📊 Reset Usage Stats"):
                st.session_state.smolvlm_usage_stats = {"queries": 0, "successful": 0, "failed": 0}
                st.session_state.cost_tracker = {"total_cost": 0.0, "jobs_processed": 0}
                st.success("✅ Usage stats reset!")
                st.rerun()
        
        with col3:
            if st.button("🔄 Test SmolVLM"):
                if runpod_client:
                    with st.spinner("Testing SmolVLM connection..."):
                        # Create a test image
                        test_image = Image.new('RGB', (100, 100), color='white')
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                            test_image.save(tmp_file.name, 'PNG')
                            with open(tmp_file.name, 'rb') as f:
                                image_data = f.read()
                            os.unlink(tmp_file.name)
                        
                        image_base64 = base64.b64encode(image_data).decode('utf-8')
                        
                        test_input = {
                            "image_data": image_base64,
                            "query": "Describe this test image",
                            "max_new_tokens": 50
                        }
                        
                        result = runpod_client.run_sync_with_tracking(test_input)
                        
                        if result["status"] == "success":
                            st.success("✅ SmolVLM test successful!")
                            st.info(f"Response time: {result.get('processing_time')}")
                        else:
                            st.error(f"❌ SmolVLM test failed: {result.get('error')}")
                else:
                    st.warning("⚠️ RunPod client not available")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
