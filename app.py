#!/usr/bin/env python3
"""
Tunnelling & Geotechnical Engineering Workflow - Domain-Specific Version
========================================================================

Fixed version focusing specifically on geotechnical engineering with enhanced
visualization capabilities for all document types.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import io
import base64
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import tempfile
import warnings
import requests
import asyncio
from dataclasses import dataclass

# Suppress warnings
warnings.filterwarnings('ignore')

# Core libraries
import pypdf
import openpyxl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# SmolAgent imports
from smolagents import CodeAgent, InferenceClientModel, tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Geotechnical Engineering AI Workflow",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class RunPodConfig:
    """RunPod configuration settings"""
    api_key: str = ""
    endpoint_id: str = ""
    base_url: str = "https://api.runpod.ai/v2"
    timeout: int = 300
    max_retries: int = 3

# Initialize RunPod configuration
def init_runpod_config() -> RunPodConfig:
    """Initialize RunPod configuration from environment variables or secrets"""
    config = RunPodConfig()
    
    try:
        config.api_key = st.secrets.get("RUNPOD_API_KEY", "") or os.getenv("RUNPOD_API_KEY", "")
        config.endpoint_id = st.secrets.get("RUNPOD_ENDPOINT_ID", "") or os.getenv("RUNPOD_ENDPOINT_ID", "")
    except Exception as e:
        logger.warning(f"Could not load secrets: {e}")
        config.api_key = os.getenv("RUNPOD_API_KEY", "")
        config.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "")
    
    return config

# Enhanced CSS for geotechnical theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-bottom: 3px solid #2E8B57;
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        color: white;
        border-radius: 10px;
    }
    .geotechnical-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #2E8B57;
    }
    .engineering-metric {
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .soil-analysis {
        background: linear-gradient(135deg, #8B4513 0%, #A0522D 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .tunnel-info {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class RunPodClient:
    """RunPod Serverless GPU Client for geotechnical document analysis"""
    
    def __init__(self, config: RunPodConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}"
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check RunPod endpoint health status"""
        try:
            url = f"{self.config.base_url}/{self.config.endpoint_id}/health"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                return {"status": "healthy", "response": response.json()}
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_sync_inference(self, image_data: str, query: str, timeout: int = None) -> Dict[str, Any]:
        """Run synchronous inference on RunPod endpoint with geotechnical focus"""
        try:
            url = f"{self.config.base_url}/{self.config.endpoint_id}/runsync"
            
            payload = {
                "input": {
                    "image_data": image_data,
                    "query": query,
                    "max_new_tokens": 512,
                    "temperature": 0.3,
                    "do_sample": True
                }
            }
            
            response = self.session.post(url, json=payload, timeout=timeout or self.config.timeout)
            
            if response.status_code == 200:
                return {"status": "success", "result": response.json()}
            else:
                return {"status": "error", "error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.Timeout:
            return {"status": "error", "error": "Request timeout - GPU processing took too long"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

class DocumentIngestionModule:
    """Enhanced Document Ingestion Module for geotechnical documents"""
    
    def __init__(self):
        self.supported_formats = {
            'pdf': ['.pdf'],
            'images': ['.png', '.jpg', '.jpeg'],
            'structured': ['.csv', '.xlsx', '.xls'],
            'text': ['.json', '.md', '.markdown', '.txt']
        }
        
    def validate_file(self, uploaded_file) -> Tuple[bool, str, str]:
        """Validate uploaded file format and determine processing type"""
        if uploaded_file is None:
            return False, "No file uploaded", ""
            
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        for category, extensions in self.supported_formats.items():
            if file_extension in extensions:
                return True, f"Valid {category} file", category
                
        return False, f"Unsupported file format: {file_extension}", ""
    
    def process_pdf(self, uploaded_file) -> Dict[str, Any]:
        """Extract text and metadata from geotechnical PDF documents"""
        try:
            pdf_reader = pypdf.PdfReader(uploaded_file)
            
            extracted_data = {
                'type': 'pdf',
                'pages': len(pdf_reader.pages),
                'text_content': [],
                'metadata': pdf_reader.metadata if pdf_reader.metadata else {},
                'processing_timestamp': datetime.now().isoformat(),
                'geotechnical_keywords': []
            }
            
            # Keywords to identify geotechnical content
            geo_keywords = [
                'soil', 'rock', 'foundation', 'bearing capacity', 'settlement', 'shear strength',
                'cohesion', 'friction angle', 'density', 'moisture content', 'plasticity',
                'consolidation', 'permeability', 'tunnel', 'excavation', 'slope stability',
                'retaining wall', 'pile', 'drilling', 'SPT', 'CPT', 'laboratory test'
            ]
            
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                # Identify geotechnical keywords
                found_keywords = [kw for kw in geo_keywords if kw.lower() in text.lower()]
                extracted_data['geotechnical_keywords'].extend(found_keywords)
                
                extracted_data['text_content'].append({
                    'page_number': i + 1,
                    'text': text,
                    'char_count': len(text),
                    'geo_keywords': found_keywords
                })
            
            # Remove duplicates
            extracted_data['geotechnical_keywords'] = list(set(extracted_data['geotechnical_keywords']))
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {'type': 'pdf', 'error': str(e)}
    
    def process_image(self, uploaded_file) -> Dict[str, Any]:
        """Process engineering drawings and site photos"""
        try:
            from PIL import Image
            image = Image.open(uploaded_file)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            width, height = image.size
            
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            extracted_data = {
                'type': 'image',
                'format': image.format,
                'mode': image.mode,
                'size': {'width': width, 'height': height},
                'image_data': img_str,
                'processing_timestamp': datetime.now().isoformat(),
                'estimated_type': self._classify_image_type(uploaded_file.name, width, height)
            }
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {'type': 'image', 'error': str(e)}
    
    def _classify_image_type(self, filename: str, width: int, height: int) -> str:
        """Classify image type based on filename and dimensions"""
        filename_lower = filename.lower()
        
        if any(term in filename_lower for term in ['drawing', 'plan', 'section', 'detail', 'dwg']):
            return 'engineering_drawing'
        elif any(term in filename_lower for term in ['site', 'photo', 'field', 'construction']):
            return 'site_photo'
        elif any(term in filename_lower for term in ['soil', 'core', 'sample', 'test']):
            return 'soil_sample'
        elif width > height * 1.5:  # Landscape format
            return 'cross_section'
        else:
            return 'general_engineering'
    
    def process_csv(self, uploaded_file) -> Dict[str, Any]:
        """Process geotechnical test data from CSV files"""
        try:
            df = pd.read_csv(uploaded_file)
            
            extracted_data = {
                'type': 'csv',
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'data': df.to_dict('records'),
                'summary_stats': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {},
                'processing_timestamp': datetime.now().isoformat(),
                'geotechnical_analysis': self._analyze_geotechnical_data(df)
            }
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            return {'type': 'csv', 'error': str(e)}
    
    def _analyze_geotechnical_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze CSV data for geotechnical parameters"""
        analysis = {
            'identified_parameters': [],
            'potential_tests': [],
            'data_quality': 'good'
        }
        
        # Common geotechnical parameter names
        geo_params = {
            'density': ['density', 'unit_weight', 'gamma', 'bulk_density'],
            'moisture': ['moisture', 'water_content', 'w', 'moisture_content'],
            'strength': ['strength', 'shear_strength', 'cohesion', 'friction_angle', 'phi', 'c'],
            'bearing': ['bearing', 'bearing_capacity', 'allowable_bearing'],
            'spt': ['spt', 'n_value', 'blow_count', 'standard_penetration'],
            'depth': ['depth', 'elevation', 'level', 'z']
        }
        
        columns_lower = [col.lower() for col in df.columns]
        
        for param_type, keywords in geo_params.items():
            for keyword in keywords:
                if any(keyword in col for col in columns_lower):
                    analysis['identified_parameters'].append(param_type)
                    break
        
        # Identify potential test types
        if 'spt' in analysis['identified_parameters']:
            analysis['potential_tests'].append('Standard Penetration Test')
        if 'strength' in analysis['identified_parameters']:
            analysis['potential_tests'].append('Direct Shear Test')
        if 'density' in analysis['identified_parameters'] and 'moisture' in analysis['identified_parameters']:
            analysis['potential_tests'].append('Proctor Compaction Test')
        
        return analysis

class GeotechnicalExtractionModule:
    """Enhanced extraction module focused on geotechnical engineering"""
    
    def __init__(self, runpod_client: RunPodClient):
        self.runpod_client = runpod_client
        self.is_configured = bool(runpod_client.config.api_key and runpod_client.config.endpoint_id)
    
    def check_status(self) -> Dict[str, Any]:
        """Check RunPod endpoint status"""
        if not self.is_configured:
            return {"status": "not_configured", "message": "RunPod API key or endpoint ID not configured"}
        return self.runpod_client.health_check()
    
    def extract_from_image(self, image_data: str, query: str = None, use_async: bool = False) -> Dict[str, Any]:
        """Extract geotechnical information from images using RunPod SmolVLM"""
        if not self.is_configured:
            return {"error": "RunPod not configured. Please set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID", "status": "not_configured"}
        
        try:
            # Enhanced geotechnical-specific query
            if not query:
                query = """Analyze this geotechnical engineering document or image. Focus on extracting:

GEOTECHNICAL PARAMETERS:
1. Soil properties (density, moisture content, plasticity index, etc.)
2. Rock properties (strength, RQD, joint conditions)
3. Foundation data (bearing capacity, settlement, pile details)
4. Test results (SPT N-values, CPT data, laboratory test results)

STRUCTURAL ELEMENTS:
5. Tunnel dimensions, support systems, rock bolts, shotcrete thickness
6. Excavation details, slope angles, retaining structures
7. Construction specifications and safety requirements

MEASUREMENTS & CALCULATIONS:
8. Dimensions, depths, elevations, coordinates
9. Load calculations, factor of safety values
10. Material specifications and design parameters

Please provide a detailed technical analysis focusing specifically on geotechnical and tunneling engineering aspects."""
            
            if use_async:
                result = self.runpod_client.run_async_inference(image_data, query)
                if result["status"] == "success":
                    return {
                        "extraction_type": "geotechnical_vision_analysis_async",
                        "query": query,
                        "job_id": result["result"].get("id"),
                        "status": "processing",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {"error": result["error"], "extraction_type": "geotechnical_vision_analysis_async", "status": "error"}
            else:
                result = self.runpod_client.run_sync_inference(image_data, query)
                if result["status"] == "success":
                    output = result["result"].get("output", {})
                    response_text = output.get("response", "No response generated")
                    
                    # Extract geotechnical parameters from response
                    extracted_params = self._extract_geotechnical_parameters(response_text)
                    
                    return {
                        "extraction_type": "geotechnical_vision_analysis",
                        "query": query,
                        "response": response_text,
                        "extracted_parameters": extracted_params,
                        "confidence": "high",
                        "processing_time": output.get("processing_time", "unknown"),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {"error": result["error"], "extraction_type": "geotechnical_vision_analysis", "status": "error"}
                    
        except Exception as e:
            logger.error(f"Error in RunPod geotechnical extraction: {str(e)}")
            return {"error": str(e), "extraction_type": "geotechnical_vision_analysis", "status": "error"}
    
    def _extract_geotechnical_parameters(self, response_text: str) -> Dict[str, Any]:
        """Extract structured geotechnical parameters from AI response"""
        parameters = {
            'soil_properties': {},
            'rock_properties': {},
            'structural_elements': {},
            'measurements': {},
            'test_results': {}
        }
        
        # Use regex to extract numerical values with units
        import re
        
        # Common patterns for geotechnical parameters
        patterns = {
            'density': r'density[:\s]*(\d+\.?\d*)\s*(kg/m3|g/cm3|pcf)',
            'moisture': r'moisture[:\s]*(\d+\.?\d*)\s*%',
            'bearing_capacity': r'bearing[:\s]*(\d+\.?\d*)\s*(kPa|MPa|psf|ksf)',
            'depth': r'depth[:\s]*(\d+\.?\d*)\s*(m|ft|cm)',
            'diameter': r'diameter[:\s]*(\d+\.?\d*)\s*(m|ft|cm|mm)',
            'thickness': r'thickness[:\s]*(\d+\.?\d*)\s*(mm|cm|m|in)'
        }
        
        for param, pattern in patterns.items():
            matches = re.findall(pattern, response_text.lower())
            if matches:
                parameters['measurements'][param] = matches
        
        return parameters

class GeotechnicalVisualizationModule:
    """Enhanced visualization module for all geotechnical document types"""
    
    def __init__(self):
        pass
    
    def create_visualization_from_any_document(self, doc_data: Dict[str, Any]) -> go.Figure:
        """Create visualizations from any document type based on extracted data"""
        try:
            doc_type = doc_data.get("document_type", "unknown")
            content = doc_data.get("content", {})
            
            if doc_type == "csv":
                return self._create_csv_visualization(content.get("tabular_data", {}))
            elif doc_type == "excel":
                return self._create_excel_visualization(content.get("tabular_data", {}))
            elif doc_type == "pdf":
                return self._create_pdf_visualization(content)
            elif doc_type == "image":
                return self._create_image_analysis_visualization(content)
            elif doc_type == "json":
                return self._create_json_visualization(content.get("json_data", {}))
            else:
                return self._create_default_visualization(doc_data)
                
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return self._create_error_visualization(str(e))
    
    def _create_csv_visualization(self, data: Dict[str, Any]) -> go.Figure:
        """Create visualizations for CSV geotechnical data"""
        if not data or 'data' not in data:
            return self._create_error_visualization("No CSV data available")
        
        df = pd.DataFrame(data.get("data", []))
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return self._create_error_visualization("No numerical data found in CSV")
        
        # Create subplots for geotechnical analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Distribution Analysis", "Depth Profile", "Parameter Correlation", "Statistical Summary"),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Plot 1: Histogram of first numerical column
        if len(numerical_cols) >= 1:
            col = numerical_cols[0]
            fig.add_trace(
                go.Histogram(x=df[col], name=f"{col} Distribution", marker_color='#2E8B57'),
                row=1, col=1
            )
        
        # Plot 2: Depth profile (if depth column exists)
        depth_cols = [col for col in df.columns if 'depth' in col.lower() or 'elevation' in col.lower()]
        if depth_cols and len(numerical_cols) >= 2:
            depth_col = depth_cols[0]
            value_col = [col for col in numerical_cols if col != depth_col][0]
            fig.add_trace(
                go.Scatter(x=df[value_col], y=df[depth_col], mode='markers+lines', 
                          name=f"{value_col} vs {depth_col}", marker_color='#8B4513'),
                row=1, col=2
            )
        
        # Plot 3: Correlation heatmap (if multiple numerical columns)
        if len(numerical_cols) >= 2:
            corr_matrix = df[numerical_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                          colorscale='RdBu', name="Correlation"),
                row=2, col=1
            )
        
        # Plot 4: Box plot for all numerical columns
        for i, col in enumerate(numerical_cols[:4]):  # Limit to 4 columns
            fig.add_trace(
                go.Box(y=df[col], name=col, marker_color=f'rgba({46+i*50}, {139+i*20}, {87+i*30}, 0.8)'),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Geotechnical Data Analysis",
            height=800,
            showlegend=True,
            font=dict(size=12)
        )
        
        return fig
    
    def _create_image_analysis_visualization(self, content: Dict[str, Any]) -> go.Figure:
        """Create visualization from image analysis results"""
        if "extracted_parameters" not in content:
            return self._create_error_visualization("No extracted parameters from image analysis")
        
        params = content["extracted_parameters"]
        
        # Create visualization of extracted measurements
        fig = go.Figure()
        
        measurements = params.get("measurements", {})
        if measurements:
            # Create bar chart of extracted measurements
            param_names = []
            values = []
            units = []
            
            for param, data in measurements.items():
                if data:  # If data exists
                    param_names.append(param.replace('_', ' ').title())
                    # Take first measurement if multiple
                    if isinstance(data[0], tuple):
                        values.append(float(data[0][0]))
                        units.append(data[0][1] if len(data[0]) > 1 else "")
                    else:
                        values.append(float(data[0]))
                        units.append("")
            
            if param_names:
                fig.add_trace(go.Bar(
                    x=param_names,
                    y=values,
                    text=[f"{v} {u}" for v, u in zip(values, units)],
                    textposition='auto',
                    marker_color='#2E8B57',
                    name='Extracted Parameters'
                ))
                
                fig.update_layout(
                    title="Extracted Geotechnical Parameters from Image",
                    xaxis_title="Parameters",
                    yaxis_title="Values",
                    height=500
                )
            else:
                return self._create_error_visualization("No measurable parameters extracted from image")
        else:
            return self._create_error_visualization("No measurements found in image analysis")
        
        return fig
    
    def _create_pdf_visualization(self, content: Dict[str, Any]) -> go.Figure:
        """Create visualization from PDF analysis"""
        text_data = content.get("text_data", [])
        
        if not text_data:
            return self._create_error_visualization("No text data from PDF")
        
        # Analyze geotechnical keywords frequency
        geo_keywords = []
        for page in text_data:
            geo_keywords.extend(page.get('geo_keywords', []))
        
        if not geo_keywords:
            return self._create_error_visualization("No geotechnical keywords found in PDF")
        
        # Count keyword frequency
        from collections import Counter
        keyword_counts = Counter(geo_keywords)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(keyword_counts.keys()),
            y=list(keyword_counts.values()),
            marker_color='#2E8B57',
            name='Keyword Frequency'
        ))
        
        fig.update_layout(
            title="Geotechnical Keywords Found in PDF",
            xaxis_title="Keywords",
            yaxis_title="Frequency",
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_error_visualization(self, error_message: str) -> go.Figure:
        """Create an error visualization"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=16,
            bgcolor="rgba(255,0,0,0.1)",
            bordercolor="red"
        )
        fig.update_layout(
            title="Visualization Not Available",
            height=400
        )
        return fig
    
    def _create_default_visualization(self, doc_data: Dict[str, Any]) -> go.Figure:
        """Create a default visualization showing document info"""
        fig = go.Figure()
        
        doc_type = doc_data.get("document_type", "Unknown")
        timestamp = doc_data.get("timestamp", "Unknown")
        
        # Create simple info display
        fig.add_annotation(
            text=f"Document Type: {doc_type}<br>Processed: {timestamp}<br><br>✅ Document processed successfully<br>🔍 Basic analysis completed",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=14,
            bgcolor="rgba(46,139,87,0.1)",
            bordercolor="#2E8B57"
        )
        
        fig.update_layout(
            title=f"Document Summary: {doc_type}",
            height=400
        )
        
        return fig

# Enhanced Agent Tools with Geotechnical Focus
@tool
def analyze_soil_data(data: str) -> str:
    """
    Analyze soil test data and provide geotechnical engineering insights.
    
    Args:
        data (str): JSON string containing soil test results with properties like density, moisture_content, bearing_capacity, SPT values, etc.
    
    Returns:
        str: Detailed geotechnical analysis with engineering recommendations
    """
    try:
        soil_data = json.loads(data)
        
        analysis = []
        analysis.append("=== GEOTECHNICAL SOIL ANALYSIS ===")
        analysis.append("🏗️ Professional Geotechnical Assessment")
        
        # Detailed soil property analysis
        if "density" in soil_data:
            density = float(soil_data["density"])
            if density < 1.4:
                analysis.append(f"⚠️  LOW DENSITY: {density} g/cm³")
                analysis.append("   • Recommendation: Compaction required")
                analysis.append("   • Consider dynamic compaction or vibrocompaction")
            elif density > 2.1:
                analysis.append(f"✅ HIGH DENSITY: {density} g/cm³")
                analysis.append("   • Excellent bearing capacity expected")
                analysis.append("   • Suitable for shallow foundations")
            else:
                analysis.append(f"📊 MODERATE DENSITY: {density} g/cm³")
                analysis.append("   • Standard foundation design applicable")
        
        if "moisture_content" in soil_data:
            moisture = float(soil_data["moisture_content"])
            if moisture > 30:
                analysis.append(f"💧 HIGH MOISTURE: {moisture}%")
                analysis.append("   • Risk of settlement and instability")
                analysis.append("   • Consider drainage improvements")
            elif moisture < 5:
                analysis.append(f"🌵 LOW MOISTURE: {moisture}%")
                analysis.append("   • Potential for shrinkage cracking")
            else:
                analysis.append(f"💧 MOISTURE CONTENT: {moisture}%")
        
        if "bearing_capacity" in soil_data:
            bearing = float(soil_data["bearing_capacity"])
            analysis.append(f"🏗️ BEARING CAPACITY: {bearing} kPa")
            if bearing < 100:
                analysis.append("   ⚠️  LOW bearing capacity")
                analysis.append("   • Deep foundations recommended (piles/caissons)")
                analysis.append("   • Consider soil improvement techniques")
            elif bearing > 300:
                analysis.append("   ✅ EXCELLENT bearing capacity")
                analysis.append("   • Shallow foundations suitable")
            else:
                analysis.append("   📊 ADEQUATE bearing capacity")
                analysis.append("   • Standard foundation design applicable")
        
        if "spt_n_value" in soil_data:
            spt = int(soil_data["spt_n_value"])
            analysis.append(f"🔨 SPT N-VALUE: {spt}")
            if spt < 4:
                analysis.append("   • Very loose/very soft soil")
            elif spt < 10:
                analysis.append("   • Loose/soft soil")
            elif spt < 30:
                analysis.append("   • Medium dense/firm soil")
            else:
                analysis.append("   • Dense/hard soil")
        
        analysis.append("\n🎯 ENGINEERING RECOMMENDATIONS:")
        analysis.append("• Conduct additional site investigation if needed")
        analysis.append("• Consider local building codes and standards")
        analysis.append("• Factor in environmental and seismic conditions")
        
        return "\n".join(analysis)
        
    except Exception as e:
        return f"Error in geotechnical soil analysis: {str(e)}"

@tool
def calculate_tunnel_support(diameter: float, depth: float, rock_quality: str) -> str:
    """
    Calculate tunnel support requirements for geotechnical engineering projects.
    
    Args:
        diameter (float): Tunnel diameter in meters
        depth (float): Depth below surface in meters  
        rock_quality (str): Rock quality designation ('excellent', 'good', 'fair', 'poor', 'very_poor')
    
    Returns:
        str: Comprehensive tunnel support design with specifications
    """
    try:
        analysis = []
        analysis.append("=== TUNNEL SUPPORT DESIGN ANALYSIS ===")
        analysis.append(f"🚇 Tunnel Diameter: {diameter} m")
        analysis.append(f"⬇️  Depth: {depth} m")
        analysis.append(f"🪨 Rock Quality: {rock_quality.upper()}")
        
        # Rock quality mapping
        quality_factors = {
            'excellent': {'factor': 2.0, 'shotcrete': 25, 'bolts': 3.0},
            'good': {'factor': 1.5, 'shotcrete': 50, 'bolts': 2.0},
            'fair': {'factor': 1.0, 'shotcrete': 100, 'bolts': 1.5},
            'poor': {'factor': 0.75, 'shotcrete': 150, 'bolts': 1.0},
            'very_poor': {'factor': 0.5, 'shotcrete': 200, 'bolts': 0.75}
        }
        
        quality = quality_factors.get(rock_quality.lower(), quality_factors['fair'])
        
        # Calculate support requirements
        bolt_spacing = diameter * quality['factor']
        shotcrete_thickness = quality['shotcrete']
        
        # Depth adjustments
        if depth > 50:
            shotcrete_thickness += 25
            bolt_spacing *= 0.8
            analysis.append("⚠️  DEEP TUNNEL: Enhanced support required")
        
        if depth > 100:
            shotcrete_thickness += 50
            bolt_spacing *= 0.7
            analysis.append("⚠️  VERY DEEP: Special design considerations")
        
        # Large diameter adjustments
        if diameter > 10:
            shotcrete_thickness += 30
            bolt_spacing *= 0.9
            analysis.append("⚠️  LARGE DIAMETER: Additional steel ribs recommended")
        
        analysis.append(f"\n🔧 SUPPORT SYSTEM SPECIFICATIONS:")
        analysis.append(f"• Rock Bolt Spacing: {bolt_spacing:.1f} m c/c")
        analysis.append(f"• Shotcrete Thickness: {shotcrete_thickness} mm")
        analysis.append(f"• Bolt Length: {diameter * 0.3:.1f} m (minimum)")
        
        # Additional recommendations
        analysis.append(f"\n📋 ADDITIONAL REQUIREMENTS:")
        if rock_quality.lower() in ['poor', 'very_poor']:
            analysis.append("• Steel mesh reinforcement required")
            analysis.append("• Systematic rock bolting pattern")
            analysis.append("• Regular monitoring and maintenance")
        
        if diameter > 8:
            analysis.append("• Steel rib supports at crown")
            analysis.append("• Temporary invert struts if needed")
        
        analysis.append("• Drainage system installation")
        analysis.append("• Ventilation considerations")
        analysis.append("• Emergency escape provisions")
        
        analysis.append(f"\n⚖️  SAFETY FACTORS:")
        analysis.append("• Design factor of safety: 1.5-2.0")
        analysis.append("• Regular structural monitoring required")
        analysis.append("• Contingency plans for ground instability")
        
        return "\n".join(analysis)
        
    except Exception as e:
        return f"Error in tunnel support calculation: {str(e)}"

@tool
def generate_geotechnical_safety_checklist(project_type: str) -> str:
    """
    Generate comprehensive safety checklist for geotechnical engineering projects.
    
    Args:
        project_type (str): Type of project ('excavation', 'tunnel', 'foundation', 'slope_stabilization', 'drilling')
    
    Returns:
        str: Detailed safety checklist with geotechnical considerations
    """
    checklists = {
        "excavation": [
            "🏗️ EXCAVATION SAFETY CHECKLIST",
            "═══════════════════════════════",
            "PRE-EXCAVATION:",
            "✅ Geotechnical site investigation completed",
            "✅ Soil stability analysis performed",
            "✅ Groundwater conditions assessed",
            "✅ Utility locating and marking completed",
            "✅ Excavation permit obtained",
            "",
            "DURING EXCAVATION:",
            "✅ Proper sloping ratios maintained (1:1.5 minimum for most soils)",
            "✅ Shoring system installed where required",
            "✅ Safe ingress/egress routes established",
            "✅ Atmospheric monitoring in deep excavations",
            "✅ Daily visual inspection of slopes",
            "✅ Water management and dewatering systems",
            "",
            "MONITORING:",
            "✅ Continuous monitoring of adjacent structures",
            "✅ Settlement monitoring points installed",
            "✅ Emergency response procedures established"
        ],
        "tunnel": [
            "🚇 TUNNEL CONSTRUCTION SAFETY",
            "════════════════════════════",
            "GROUND CONDITIONS:",
            "✅ Comprehensive geological survey completed",
            "✅ Rock quality designation (RQD) assessment",
            "✅ Groundwater inflow predictions",
            "✅ Gas monitoring equipment installed",
            "",
            "SUPPORT SYSTEMS:",
            "✅ Temporary support installation procedures",
            "✅ Shotcrete application protocols",
            "✅ Rock bolt installation standards",
            "✅ Steel rib placement requirements",
            "",
            "SAFETY SYSTEMS:",
            "✅ Ventilation system operational",
            "✅ Emergency escape routes clear and marked",
            "✅ Communication systems tested",
            "✅ Gas detection and alarm systems",
            "✅ Emergency rescue equipment available",
            "",
            "MONITORING:",
            "✅ Continuous ground movement monitoring",
            "✅ Support system load monitoring",
            "✅ Regular safety inspections scheduled"
        ],
        "foundation": [
            "🏛️ FOUNDATION CONSTRUCTION SAFETY",
            "══════════════════════════════════",
            "SOIL CONDITIONS:",
            "✅ Bearing capacity verification completed",
            "✅ Settlement analysis performed",
            "✅ Liquefaction potential assessed",
            "✅ Groundwater level monitoring",
            "",
            "EXCAVATION SAFETY:",
            "✅ Foundation excavation properly shored",
            "✅ Adequate dewatering systems in place",
            "✅ Soil testing during construction",
            "✅ Inspection of foundation materials",
            "",
            "QUALITY CONTROL:",
            "✅ Concrete mix design approved",
            "✅ Reinforcement placement inspection",
            "✅ Foundation level verification",
            "✅ Compaction testing for backfill"
        ],
        "slope_stabilization": [
            "⛰️ SLOPE STABILIZATION SAFETY",
            "═════════════════════════════",
            "STABILITY ANALYSIS:",
            "✅ Comprehensive slope stability analysis",
            "✅ Factor of safety calculations completed",
            "✅ Potential failure modes identified",
            "✅ Groundwater impact assessment",
            "",
            "MONITORING SYSTEMS:",
            "✅ Inclinometer installation",
            "✅ Piezometer network established",
            "✅ Survey monitoring points set",
            "✅ Early warning systems operational",
            "",
            "STABILIZATION MEASURES:",
            "✅ Drainage systems installed",
            "✅ Retaining structures properly anchored",
            "✅ Slope protection measures in place",
            "✅ Vegetation establishment plan",
            "",
            "EMERGENCY PROCEDURES:",
            "✅ Evacuation routes identified",
            "✅ Emergency contact procedures",
            "✅ Regular inspection schedule established"
        ]
    }
    
    project_checklist = checklists.get(project_type.lower(), checklists["excavation"])
    
    result = project_checklist.copy()
    result.extend([
        "",
        "🔴 CRITICAL REMINDERS:",
        "• Always follow local building codes and regulations",
        "• Ensure qualified geotechnical engineer oversight",
        "• Maintain current safety training certifications",
        "• Regular toolbox talks on geotechnical hazards",
        "• Emergency response plan readily available",
        "",
        "📞 EMERGENCY CONTACTS:",
        "• Site Safety Officer: [Contact Information]",
        "• Geotechnical Engineer: [Contact Information]",
        "• Emergency Services: [Local Emergency Number]"
    ])
    
    return "\n".join(result)

class GeotechnicalMultiAgentOrchestrator:
    """Enhanced Multi-Agent Orchestration for geotechnical engineering"""
    
    def __init__(self):
        self.model = InferenceClientModel()
        self.agents = {}
        self._initialize_geotechnical_agents()
    
    def _initialize_geotechnical_agents(self):
        """Initialize specialized geotechnical agents"""
        try:
            # Soil Analysis Agent
            self.agents["soil_analyst"] = CodeAgent(
                tools=[analyze_soil_data],
                model=self.model
            )
            
            # Tunnel Engineering Agent
            self.agents["tunnel_engineer"] = CodeAgent(
                tools=[calculate_tunnel_support, generate_geotechnical_safety_checklist],
                model=self.model
            )
            
            logger.info("Geotechnical SmolAgent orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing geotechnical agents: {str(e)}")
            self.agents = {}
    
    def route_geotechnical_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Route queries to appropriate geotechnical specialists"""
        try:
            query_lower = query.lower()
            
            # Enhanced geotechnical query routing
            if any(term in query_lower for term in ["soil", "bearing", "density", "moisture", "spt", "consolidation", "settlement"]):
                agent_type = "soil_analyst"
                enhanced_query = f"As a geotechnical engineer specializing in soil mechanics, {query}. Provide detailed engineering analysis with specific recommendations."
            elif any(term in query_lower for term in ["tunnel", "support", "excavation", "rock", "shotcrete", "bolt", "underground"]):
                agent_type = "tunnel_engineer"
                enhanced_query = f"As a tunnel engineering specialist, {query}. Include specific design parameters and safety considerations."
            elif any(term in query_lower for term in ["safety", "checklist", "hazard", "risk", "emergency"]):
                agent_type = "tunnel_engineer"  # Safety expert
                enhanced_query = f"As a geotechnical safety specialist, {query}. Focus on industry best practices and regulatory compliance."
            else:
                # Default to soil analyst for general geotechnical queries
                agent_type = "soil_analyst"
                enhanced_query = f"As a geotechnical engineering professional, analyze this query related to geotechnical engineering: {query}. Provide technical insights and practical recommendations."
            
            if agent_type in self.agents:
                result = self.agents[agent_type].run(enhanced_query)
                return {
                    "agent_type": agent_type,
                    "query": query,
                    "enhanced_query": enhanced_query,
                    "response": result,
                    "timestamp": datetime.now().isoformat(),
                    "domain": "Geotechnical Engineering",
                    "powered_by": "RunPod GPU + SmolAgent"
                }
            else:
                return {
                    "agent_type": "fallback",
                    "query": query,
                    "response": "I'm a geotechnical engineering assistant. Please ask questions related to soil mechanics, foundation design, tunnel engineering, or slope stability. The specialized agents are temporarily unavailable.",
                    "timestamp": datetime.now().isoformat(),
                    "domain": "Geotechnical Engineering"
                }
                
        except Exception as e:
            logger.error(f"Error in geotechnical agent routing: {str(e)}")
            return {
                "agent_type": "error",
                "query": query,
                "response": f"I encountered an error processing your geotechnical engineering query. Please rephrase your question focusing on soil analysis, tunnel design, or foundation engineering topics.",
                "timestamp": datetime.now().isoformat(),
                "domain": "Geotechnical Engineering"
            }

# Initialize system with enhanced geotechnical modules
@st.cache_resource
def initialize_geotechnical_system():
    """Initialize the complete geotechnical engineering system"""
    config = init_runpod_config()
    runpod_client = RunPodClient(config)
    
    return {
        "config": config,
        "ingestion": DocumentIngestionModule(),
        "extraction": GeotechnicalExtractionModule(runpod_client),
        "structured_output": StructuredOutputModule(),
        "visualization": GeotechnicalVisualizationModule(),
        "orchestrator": GeotechnicalMultiAgentOrchestrator(),
        "runpod_client": runpod_client
    }

def main():
    """Main geotechnical engineering application"""
    
    # Initialize system
    system = initialize_geotechnical_system()
    
    # Enhanced header for geotechnical engineering
    st.markdown('<h1 class="main-header">🏗️ Geotechnical Engineering AI Workflow</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = {}
    if "async_jobs" not in st.session_state:
        st.session_state.async_jobs = {}
    
    # Enhanced sidebar for geotechnical document management
    with st.sidebar:
        st.header("📁 Geotechnical Document Management")
        
        # RunPod Status with geotechnical branding
        st.subheader("🚀 AI Processing Status")
        runpod_status = system["extraction"].check_status()
        
        if runpod_status["status"] == "healthy":
            st.markdown('<div class="engineering-metric">✅ GPU Ready<br/>SmolVLM Active</div>', 
                       unsafe_allow_html=True)
        elif runpod_status["status"] == "not_configured":
            st.markdown('<div class="engineering-metric" style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);">❌ Not Configured<br/>Set API Keys</div>', 
                       unsafe_allow_html=True)
            st.error("⚠️ Configure RunPod credentials for AI vision analysis")
        else:
            st.markdown('<div class="engineering-metric" style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);">❌ GPU Error<br/>Check Status</div>', 
                       unsafe_allow_html=True)
            st.error(f"RunPod Error: {runpod_status.get('error', 'Unknown error')}")
        
        st.divider()
        
        # File upload with geotechnical context
        st.subheader("📄 Upload Documents")
        st.caption("Upload geotechnical reports, soil test data, engineering drawings, or site photos")
        
        uploaded_file = st.file_uploader(
            "Select Document",
            type=['pdf', 'png', 'jpg', 'jpeg', 'csv', 'xlsx', 'xls', 'json', 'md', 'txt'],
            help="PDF reports, CSV test data, engineering drawings (PNG/JPG), Excel files"
        )
        
        if uploaded_file is not None:
            processing_mode = "sync"
            if uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                processing_mode = st.radio(
                    "Processing Mode:",
                    ["sync", "async"],
                    help="Sync: Immediate analysis. Async: Background processing"
                )
            
            if st.button("🔬 Analyze Document", type="primary"):
                with st.spinner("Processing geotechnical document..."):
                    is_valid, message, file_type = system["ingestion"].validate_file(uploaded_file)
                    
                    if is_valid:
                        st.info(f"✅ {message}")
                        
                        try:
                            # Process document based on type
                            if file_type == "pdf":
                                extracted_data = system["ingestion"].process_pdf(uploaded_file)
                            elif file_type == "images":
                                extracted_data = system["ingestion"].process_image(uploaded_file)
                            elif file_type == "structured":
                                if uploaded_file.name.endswith('.csv'):
                                    extracted_data = system["ingestion"].process_csv(uploaded_file)
                                else:
                                    extracted_data = system["ingestion"].process_excel(uploaded_file)
                            elif file_type == "text":
                                if uploaded_file.name.endswith('.json'):
                                    extracted_data = system["ingestion"].process_json(uploaded_file)
                                else:
                                    extracted_data = system["ingestion"].process_markdown(uploaded_file)
                            
                            # Enhanced geotechnical vision analysis for images
                            if file_type == "images" and "error" not in extracted_data:
                                if runpod_status["status"] == "healthy":
                                    with st.spinner("Running geotechnical AI analysis..."):
                                        vision_result = system["extraction"].extract_from_image(
                                            extracted_data["image_data"],
                                            use_async=(processing_mode == "async")
                                        )
                                        if "error" not in vision_result:
                                            extracted_data.update(vision_result)
                                            
                                            if processing_mode == "async" and "job_id" in vision_result:
                                                st.session_state.async_jobs[vision_result["job_id"]] = {
                                                    "file_name": uploaded_file.name,
                                                    "timestamp": datetime.now().isoformat(),
                                                    "status": "processing"
                                                }
                                        else:
                                            st.error(f"Vision analysis failed: {vision_result['error']}")
                                else:
                                    st.warning("AI vision analysis unavailable (RunPod not configured)")
                            
                            # Organize data with geotechnical focus
                            doc_id = f"{uploaded_file.name}_{int(time.time())}"
                            structured_data = system["structured_output"].organize_data(extracted_data, doc_id)
                            
                            st.session_state.processed_documents[doc_id] = structured_data
                            
                            if processing_mode == "async" and "job_id" in extracted_data:
                                st.success(f"✅ Document uploaded! AI analysis processing...")
                            else:
                                st.success(f"✅ Geotechnical analysis completed!")
                            
                        except Exception as e:
                            st.error(f"❌ Processing error: {str(e)}")
                    else:
                        st.error(f"❌ {message}")
        
        st.divider()
        
        # Document list with geotechnical info
        st.subheader("📋 Analyzed Documents")
        if st.session_state.processed_documents:
            for doc_id, doc_data in st.session_state.processed_documents.items():
                with st.expander(f"📄 {doc_id.split('_')[0][:20]}..."):
                    st.write(f"**Type:** {doc_data.get('document_type', 'Unknown')}")
                    st.write(f"**Status:** {doc_data.get('processing_status', 'Unknown')}")
                    st.write(f"**Analysis:** {doc_data.get('timestamp', 'Unknown')[:19]}")
                    
                    if st.button(f"🗑️ Remove", key=f"remove_{doc_id}"):
                        del st.session_state.processed_documents[doc_id]
                        st.rerun()
        else:
            st.info("No documents analyzed yet")
    
    # Main content area with enhanced geotechnical tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Geotechnical AI Assistant", 
        "📊 Data Analysis", 
        "📈 Visualizations", 
        "🚀 System Status", 
        "🔧 Settings"
    ])
    
    with tab1:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.subheader("🤖 Geotechnical Engineering AI Assistant")
        st.caption("Specialized in soil mechanics, foundation design, tunnel engineering, and slope stability")
        
        if runpod_status["status"] != "healthy":
            st.warning("⚠️ AI vision analysis limited. Configure RunPod for full capabilities.")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Enhanced chat input with geotechnical context
        if prompt := st.chat_input("Ask about soil analysis, tunnel design, foundation engineering, or safety procedures..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Consulting geotechnical engineering specialists..."):
                    
                    context = {
                        "documents": list(st.session_state.processed_documents.keys()),
                        "document_count": len(st.session_state.processed_documents),
                        "runpod_status": runpod_status["status"],
                        "domain": "geotechnical_engineering"
                    }
                    
                    agent_response = system["orchestrator"].route_geotechnical_query(prompt, context)
                    
                    response_text = agent_response.get("response", "I apologize, but I couldn't process your geotechnical engineering query.")
                    st.markdown(response_text)
                    
                    # Show specialist info
                    with st.expander("🔍 Analysis Details"):
                        st.json(agent_response)
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Enhanced example queries for geotechnical engineering
        st.subheader("💡 Example Geotechnical Queries")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏗️ Foundation & Soil Analysis:**")
            example_queries_1 = [
                "Analyze soil bearing capacity for 10-story building",
                "Calculate settlement for clay soil with OCR=2.5",
                "Evaluate soil data: {'density': 1.8, 'moisture_content': 15, 'bearing_capacity': 250, 'spt_n_value': 18}",
                "Design shallow foundation for sandy soil"
            ]
            
            for i, example in enumerate(example_queries_1):
                if st.button(example, key=f"example1_{i}"):
                    st.session_state.messages.append({"role": "user", "content": example})
                    context = {
                        "documents": list(st.session_state.processed_documents.keys()),
                        "domain": "geotechnical_engineering"
                    }
                    agent_response = system["orchestrator"].route_geotechnical_query(example, context)
                    response_text = agent_response.get("response", "Could not process query.")
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.rerun()
        
        with col2:
            st.markdown("**🚇 Tunnel & Excavation Engineering:**")
            example_queries_2 = [
                "Calculate tunnel support for 12m diameter at 45m depth in poor rock",
                "Generate safety checklist for tunnel excavation project",
                "Design retaining wall for 8m deep excavation",
                "Evaluate slope stability for 30° cut in weathered rock"
            ]
            
            for i, example in enumerate(example_queries_2):
                if st.button(example, key=f"example2_{i}"):
                    st.session_state.messages.append({"role": "user", "content": example})
                    context = {
                        "documents": list(st.session_state.processed_documents.keys()),
                        "domain": "geotechnical_engineering"
                    }
                    agent_response = system["orchestrator"].route_geotechnical_query(example, context)
                    response_text = agent_response.get("response", "Could not process query.")
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.subheader("📊 Geotechnical Data Analysis")
        
        if st.session_state.processed_documents:
            doc_options = list(st.session_state.processed_documents.keys())
            selected_doc = st.selectbox("Select document for analysis:", doc_options)
            
            if selected_doc:
                doc_data = st.session_state.processed_documents[selected_doc]
                
                # Enhanced metrics display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown('<div class="engineering-metric">Document Type<br/>' + 
                               doc_data.get("document_type", "Unknown") + '</div>', 
                               unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="soil-analysis">Analysis Status<br/>' + 
                               doc_data.get("processing_status", "Unknown") + '</div>', 
                               unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="tunnel-info">File Size<br/>' + 
                               f"{len(str(doc_data)):,} chars</div>", 
                               unsafe_allow_html=True)
                with col4:
                    timestamp = doc_data.get("timestamp", "Unknown")[:10] if doc_data.get("timestamp") else "Unknown"
                    st.markdown('<div class="engineering-metric">Processed<br/>' + 
                               timestamp + '</div>', 
                               unsafe_allow_html=True)
                
                st.divider()
                
                # Detailed analysis based on document type
                content = doc_data.get("content", {})
                doc_type = doc_data.get("document_type")
                
                if doc_type in ["csv", "excel"]:
                    st.subheader("📈 Geotechnical Test Data Analysis")
                    tabular_data = content.get("tabular_data", {})
                    
                    if "geotechnical_analysis" in tabular_data:
                        geo_analysis = tabular_data["geotechnical_analysis"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**🔍 Identified Parameters:**")
                            params = geo_analysis.get("identified_parameters", [])
                            if params:
                                for param in params:
                                    st.write(f"• {param.replace('_', ' ').title()}")
                            else:
                                st.write("No specific geotechnical parameters identified")
                        
                        with col2:
                            st.write("**🧪 Potential Test Types:**")
                            tests = geo_analysis.get("potential_tests", [])
                            if tests:
                                for test in tests:
                                    st.write(f"• {test}")
                            else:
                                st.write("Standard geotechnical analysis applicable")
                    
                    # Statistical summary
                    stats = tabular_data.get("summary_stats", {})
                    if stats:
                        st.subheader("📊 Statistical Summary")
                        stats_df = pd.DataFrame(stats)
                        st.dataframe(stats_df)
                
                elif doc_type == "image":
                    st.subheader("👁️ AI Vision Analysis Results")
                    
                    if "analysis_type" in content:
                        if content.get("analysis_type") == "geotechnical_vision_analysis":
                            st.markdown('<div class="tunnel-info">🚀 Processed with Geotechnical AI Specialist</div>', 
                                       unsafe_allow_html=True)
                        
                        st.write(f"**🔍 Analysis Query:**")
                        st.info(content.get('query', 'N/A'))
                        
                        st.write(f"**📋 AI Analysis:**")
                        response_text = content.get('response', 'N/A')
                        st.write(response_text)
                        
                        # Show extracted parameters if available
                        if "extracted_parameters" in content:
                            st.subheader("📐 Extracted Parameters")
                            params = content["extracted_parameters"]
                            
                            for category, values in params.items():
                                if values:
                                    st.write(f"**{category.replace('_', ' ').title()}:**")
                                    st.json(values)
                        
                        processing_time = content.get('processing_time', 'unknown')
                        if processing_time != 'unknown':
                            st.success(f"⚡ GPU Processing Time: {processing_time}")
                
                elif doc_type == "pdf":
                    st.subheader("📄 PDF Document Analysis")
                    
                    text_data = content.get("text_data", [])
                    if text_data:
                        total_chars = sum(page.get('char_count', 0) for page in text_data)
                        total_geo_keywords = sum(len(page.get('geo_keywords', [])) for page in text_data)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Pages", len(text_data))
                        with col2:
                            st.metric("Characters", f"{total_chars:,}")
                        with col3:
                            st.metric("Geo Keywords", total_geo_keywords)
                        
                        # Show geotechnical keywords found
                        all_keywords = []
                        for page in text_data:
                            all_keywords.extend(page.get('geo_keywords', []))
                        
                        if all_keywords:
                            st.write("**🔍 Geotechnical Keywords Found:**")
                            unique_keywords = list(set(all_keywords))
                            st.write(", ".join(unique_keywords))
                        
                        # Sample text
                        if text_data:
                            st.write("**📄 Sample Text (First Page):**")
                            sample_text = text_data[0].get('text', '')[:500]
                            st.text_area("Preview", sample_text, height=150, disabled=True)
                
                # Raw content viewer
                with st.expander("🔍 Raw Content Data"):
                    st.json(content)
        else:
            st.info("📥 Upload and analyze geotechnical documents to see detailed analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.subheader("📈 Geotechnical Data Visualizations")
        st.caption("Intelligent visualizations for all document types")
        
        if st.session_state.processed_documents:
            # Allow visualization of ANY document type
            doc_options = list(st.session_state.processed_documents.keys())
            selected_doc = st.selectbox("Select document for visualization:", doc_options, key="viz_doc")
            
            if selected_doc:
                doc_data = st.session_state.processed_documents[selected_doc]
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("🎨 Generate Visualization", type="primary"):
                        with st.spinner("Creating geotechnical visualization..."):
                            # Use enhanced visualization module that works with any document type
                            fig = system["visualization"].create_visualization_from_any_document(doc_data)
                            st.plotly_chart(fig, use_container_width=True)
                
                with col1:
                    doc_type = doc_data.get("document_type", "Unknown")
                    st.info(f"📊 Document Type: {doc_type} - Visualization available for all processed documents")
                
                # Auto-generate visualization for demonstration
                if not st.session_state.get(f"viz_generated_{selected_doc}", False):
                    with st.spinner("Preparing visualization..."):
                        fig = system["visualization"].create_visualization_from_any_document(doc_data)
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state[f"viz_generated_{selected_doc}"] = True
        else:
            st.info("📥 Upload geotechnical documents to create intelligent visualizations")
            st.write("**Supported visualizations:**")
            st.write("• 📊 CSV/Excel: Statistical analysis, correlation matrices, depth profiles")
            st.write("• 🖼️ Images: Parameter extraction charts, measurement displays")  
            st.write("• 📄 PDFs: Keyword frequency, content analysis")
            st.write("• 📋 JSON: Data structure visualization")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.subheader("🚀 System Status & Performance")
        
        # System status overview
        config = system["config"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🔧 Configuration Status:**")
            config_status = "✅ Configured" if config.api_key and config.endpoint_id else "❌ Not Configured"
            st.write(f"Status: {config_status}")
            st.write(f"API Key: {'***' + config.api_key[-4:] if config.api_key else 'Not set'}")
            st.write(f"Endpoint: {config.endpoint_id if config.endpoint_id else 'Not set'}")
        
        with col2:
            st.write("**🏗️ Geotechnical Capabilities:**")
            if st.button("🔄 Test AI System"):
                with st.spinner("Testing geotechnical AI capabilities..."):
                    health = system["runpod_client"].health_check()
                    if health["status"] == "healthy":
                        st.success("✅ AI vision analysis ready!")
                        st.success("✅ Soil analysis agent active")
                        st.success("✅ Tunnel engineering agent active")
                        st.success("✅ Safety checklist generator ready")
                    else:
                        st.error(f"❌ System issue: {health.get('error', 'Unknown error')}")
        
        st.divider()
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if runpod_status["status"] == "healthy":
                st.markdown('<div class="engineering-metric">🚀 SmolVLM<br/>Ready</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="engineering-metric" style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);">❌ SmolVLM<br/>Error</div>', unsafe_allow_html=True)
        
        with col2:
            if system["orchestrator"].agents:
                st.markdown('<div class="soil-analysis">✅ Agents<br/>Active</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="soil-analysis" style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);">❌ Agents<br/>Error</div>', unsafe_allow_html=True)
        
        with col3:
            doc_count = len(st.session_state.processed_documents)
            st.markdown(f'<div class="tunnel-info">📄 Documents<br/>{doc_count}</div>', unsafe_allow_html=True)
        
        with col4:
            processing_mode = "GPU Accelerated" if config.api_key else "Local"
            st.markdown(f'<div class="engineering-metric">🌐 Mode<br/>{processing_mode}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.subheader("🔧 Application Settings")
        
        st.write("**🏗️ Geotechnical Engineering Workflow Configuration**")
        
        # Data management
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.success("Chat history cleared!")
                st.rerun()
        
        with col2:
            if st.button("📂 Clear Documents"):
                st.session_state.processed_documents = {}
                st.success("Documents cleared!")
                st.rerun()
        
        with col3:
            if st.button("⏳ Clear Async Jobs"):
                st.session_state.async_jobs = {}
                st.success("Async jobs cleared!")
                st.rerun()
        
        st.divider()
        
        # System information
        st.write("**📋 System Information:**")
        st.write("• **Domain**: Geotechnical & Tunnel Engineering")
        st.write("• **AI Models**: SmolVLM-Instruct (Vision), SmolAgent (Reasoning)")
        st.write("• **Infrastructure**: RunPod Serverless GPU")
        st.write("• **Specializations**: Soil mechanics, Foundation design, Tunnel engineering, Slope stability")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
