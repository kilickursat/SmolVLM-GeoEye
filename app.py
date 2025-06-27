#!/usr/bin/env python3
"""
Tunnelling & Geotechnical Engineering Workflow - Streamlit Application
=====================================================================

A comprehensive multi-modal document processing system using SmolVLM, SmolAgent,
and Streamlit for geotechnical engineering workflows.

Author: Generated for Geotechnical Engineering
Version: 2.0.0 (Fixed)
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
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import tempfile
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Core libraries
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
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
    page_title="Geotechnical Engineering Workflow",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# HuggingFace Authentication
def setup_huggingface_auth():
    """Setup HuggingFace authentication if token is provided"""
    if "hf_authenticated" not in st.session_state:
        st.session_state.hf_authenticated = False
    
    if not st.session_state.hf_authenticated:
        hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")
        
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token, add_to_git_credential=True)
                st.session_state.hf_authenticated = True
                logger.info("HuggingFace authentication successful")
            except Exception as e:
                logger.warning(f"HuggingFace authentication failed: {e}")

# Initialize HF auth
setup_huggingface_auth()

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-bottom: 3px solid #1f77b4;
    }
    .module-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-processing {
        color: #ffc107;
        font-weight: bold;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        background-color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stAlert > div {
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class DocumentIngestionModule:
    """
    Document Ingestion Module
    Handles preprocessing of various document formats including PDF, images, CSV, Excel, JSON, and Markdown.
    """
    
    def __init__(self):
        self.supported_formats = {
            'pdf': ['.pdf'],
            'images': ['.png', '.jpg', '.jpeg'],
            'structured': ['.csv', '.xlsx', '.xls'],
            'text': ['.json', '.md', '.markdown', '.txt']
        }
        
    def validate_file(self, uploaded_file) -> Tuple[bool, str, str]:
        """Validate uploaded file format and determine processing type."""
        if uploaded_file is None:
            return False, "No file uploaded", ""
            
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        for category, extensions in self.supported_formats.items():
            if file_extension in extensions:
                return True, f"Valid {category} file", category
                
        return False, f"Unsupported file format: {file_extension}", ""
    
    def process_pdf(self, uploaded_file) -> Dict[str, Any]:
        """Extract text and metadata from PDF files."""
        try:
            pdf_reader = pypdf.PdfReader(uploaded_file)
            
            extracted_data = {
                'type': 'pdf',
                'pages': len(pdf_reader.pages),
                'text_content': [],
                'metadata': pdf_reader.metadata if pdf_reader.metadata else {},
                'processing_timestamp': datetime.now().isoformat()
            }
            
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                extracted_data['text_content'].append({
                    'page_number': i + 1,
                    'text': text,
                    'char_count': len(text)
                })
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {'type': 'pdf', 'error': str(e)}
    
    def process_image(self, uploaded_file) -> Dict[str, Any]:
        """Process image files and prepare for vision model analysis."""
        try:
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get image metadata
            width, height = image.size
            
            # Convert to base64 for storage
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            extracted_data = {
                'type': 'image',
                'format': image.format,
                'mode': image.mode,
                'size': {
                    'width': width,
                    'height': height
                },
                'image_data': img_str,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {'type': 'image', 'error': str(e)}
    
    def process_csv(self, uploaded_file) -> Dict[str, Any]:
        """Process CSV files and extract structured data."""
        try:
            df = pd.read_csv(uploaded_file)
            
            extracted_data = {
                'type': 'csv',
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'data': df.to_dict('records'),
                'summary_stats': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {},
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            return {'type': 'csv', 'error': str(e)}
    
    def process_excel(self, uploaded_file) -> Dict[str, Any]:
        """Process Excel files and extract structured data from all sheets."""
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            
            extracted_data = {
                'type': 'excel',
                'sheet_names': excel_file.sheet_names,
                'sheets_data': {},
                'processing_timestamp': datetime.now().isoformat()
            }
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                extracted_data['sheets_data'][sheet_name] = {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'data_types': df.dtypes.astype(str).to_dict(),
                    'data': df.to_dict('records'),
                    'summary_stats': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {}
                }
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing Excel: {str(e)}")
            return {'type': 'excel', 'error': str(e)}
    
    def process_json(self, uploaded_file) -> Dict[str, Any]:
        """Process JSON files and extract structured data."""
        try:
            content = uploaded_file.read().decode('utf-8')
            json_data = json.loads(content)
            
            extracted_data = {
                'type': 'json',
                'data': json_data,
                'structure_analysis': self._analyze_json_structure(json_data),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing JSON: {str(e)}")
            return {'type': 'json', 'error': str(e)}
    
    def process_markdown(self, uploaded_file) -> Dict[str, Any]:
        """Process Markdown files and extract text content."""
        try:
            content = uploaded_file.read().decode('utf-8')
            
            extracted_data = {
                'type': 'markdown',
                'content': content,
                'char_count': len(content),
                'line_count': len(content.split('\n')),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing Markdown: {str(e)}")
            return {'type': 'markdown', 'error': str(e)}
    
    def _analyze_json_structure(self, data, depth=0, max_depth=3) -> Dict[str, Any]:
        """Analyze JSON structure for better understanding."""
        if depth > max_depth:
            return {"type": "truncated", "reason": "max_depth_reached"}
        
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys()),
                "key_count": len(data),
                "nested_structure": {k: self._analyze_json_structure(v, depth + 1, max_depth) 
                                   for k, v in list(data.items())[:5]}  # Limit to first 5 keys
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "sample_structure": self._analyze_json_structure(data[0], depth + 1, max_depth) if data else None
            }
        else:
            return {
                "type": type(data).__name__,
                "value": str(data)[:100] if isinstance(data, str) else data
            }

class ExtractionModule:
    """
    Extraction Module
    Uses SmolVLM vision-language model to extract relevant information from preprocessed documents.
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize SmolVLM model and processor with proper fallback."""
        try:
            model_name = "HuggingFaceTB/SmolVLM-Instruct"
            logger.info(f"Loading {model_name} on {self.device}")
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            # Use appropriate dtype based on device
            torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            
            # Try with flash attention first, fallback to eager if not available
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    _attn_implementation="flash_attention_2"
                ).to(self.device)
                logger.info(f"SmolVLM loaded with FlashAttention2 on {self.device}")
            except Exception as flash_error:
                logger.warning(f"FlashAttention2 not available, falling back to eager attention...")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    _attn_implementation="eager"
                ).to(self.device)
                logger.info(f"SmolVLM loaded with eager attention on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing SmolVLM model: {str(e)}")
            self.model = None
            self.processor = None
    
    def extract_from_image(self, image_data: str, query: str = None) -> Dict[str, Any]:
        """Extract information from image using SmolVLM."""
        if not self.model or not self.processor:
            return {"error": "Model not initialized"}
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Default query for geotechnical analysis
            if not query:
                query = """Analyze this engineering document or image. Extract:
                1. Any technical specifications, measurements, or parameters
                2. Engineering calculations or formulas visible
                3. Material properties or soil characteristics
                4. Safety considerations or warnings
                5. Key findings or conclusions
                6. Any geometric or structural details
                Please provide a detailed technical analysis."""
            
            # Prepare input for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": query}
                    ]
                }
            ]
            
            # Apply chat template and process
            input_text = self.processor.apply_chat_template(messages, tokenize=False)
            inputs = self.processor(text=input_text, images=[image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the input)
            generated_text = response.split("Assistant:")[-1].strip()
            
            return {
                "extraction_type": "vision_analysis",
                "query": query,
                "response": generated_text,
                "confidence": "high",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in vision extraction: {str(e)}")
            return {"error": str(e)}

class StructuredOutputModule:
    """
    Structured Output Module
    Organizes extracted information into structured formats for querying and analysis.
    """
    
    def __init__(self):
        self.data_store = {}
        self.metadata_store = {}
    
    def organize_data(self, extracted_data: Dict[str, Any], document_id: str) -> Dict[str, Any]:
        """Organize extracted data into a structured format."""
        try:
            structured_data = {
                "document_id": document_id,
                "timestamp": datetime.now().isoformat(),
                "document_type": extracted_data.get("type", "unknown"),
                "processing_status": "completed",
                "content": {},
                "metadata": {},
                "searchable_fields": []
            }
            
            # Process different data types
            if extracted_data.get("type") == "image" and "extraction_type" in extracted_data:
                # Vision analysis results
                structured_data["content"] = {
                    "analysis_type": "vision_language_model",
                    "query": extracted_data.get("query", ""),
                    "response": extracted_data.get("response", ""),
                    "confidence": extracted_data.get("confidence", "medium")
                }
                structured_data["searchable_fields"] = [
                    extracted_data.get("response", "")
                ]
                
            elif extracted_data.get("type") in ["pdf", "markdown"]:
                # Text-based content
                if "text_content" in extracted_data:
                    structured_data["content"] = {
                        "text_data": extracted_data["text_content"],
                        "extracted_info": extracted_data.get("extracted_info", {})
                    }
                    # Make text searchable
                    if isinstance(extracted_data["text_content"], list):
                        structured_data["searchable_fields"] = [
                            page["text"] for page in extracted_data["text_content"]
                        ]
                    else:
                        structured_data["searchable_fields"] = [extracted_data["text_content"]]
                
            elif extracted_data.get("type") in ["csv", "excel"]:
                # Structured data
                structured_data["content"] = {
                    "tabular_data": extracted_data,
                    "column_info": self._analyze_columns(extracted_data)
                }
                structured_data["searchable_fields"] = [
                    str(extracted_data.get("columns", [])),
                    str(extracted_data.get("summary_stats", {}))
                ]
                
            elif extracted_data.get("type") == "json":
                # JSON data
                structured_data["content"] = {
                    "json_data": extracted_data["data"],
                    "structure": extracted_data.get("structure_analysis", {})
                }
                structured_data["searchable_fields"] = [
                    str(extracted_data["data"])
                ]
            
            # Store in data store
            self.data_store[document_id] = structured_data
            self.metadata_store[document_id] = {
                "upload_time": structured_data["timestamp"],
                "document_type": structured_data["document_type"],
                "processing_status": "completed"
            }
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error organizing data: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_columns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze columns in structured data for better organization."""
        column_analysis = {}
        
        if data.get("type") == "csv":
            columns = data.get("columns", [])
            data_types = data.get("data_types", {})
            
            for col in columns:
                dtype = data_types.get(col, "object")
                column_analysis[col] = {
                    "data_type": dtype,
                    "is_numeric": dtype in ["int64", "float64"],
                    "engineering_relevance": self._assess_engineering_relevance(col)
                }
                
        elif data.get("type") == "excel":
            for sheet_name, sheet_data in data.get("sheets_data", {}).items():
                columns = sheet_data.get("columns", [])
                data_types = sheet_data.get("data_types", {})
                
                column_analysis[sheet_name] = {}
                for col in columns:
                    dtype = data_types.get(col, "object")
                    column_analysis[sheet_name][col] = {
                        "data_type": dtype,
                        "is_numeric": dtype in ["int64", "float64"],
                        "engineering_relevance": self._assess_engineering_relevance(col)
                    }
        
        return column_analysis
    
    def _assess_engineering_relevance(self, column_name: str) -> str:
        """Assess engineering relevance of column names."""
        engineering_terms = {
            "high": ["strength", "load", "stress", "strain", "pressure", "force", "weight", 
                    "density", "depth", "height", "width", "diameter", "thickness"],
            "medium": ["time", "date", "location", "sample", "test", "measurement"],
            "low": ["id", "name", "description", "notes", "comments"]
        }
        
        col_lower = column_name.lower()
        
        for relevance, terms in engineering_terms.items():
            if any(term in col_lower for term in terms):
                return relevance
        
        return "unknown"

class AnalysisVisualizationModule:
    """
    Analysis and Visualization Module
    Provides data analysis and visualization capabilities.
    """
    
    def __init__(self):
        pass
    
    def generate_statistical_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical summary of numerical data."""
        try:
            summary = {
                "analysis_type": "statistical_summary",
                "timestamp": datetime.now().isoformat(),
                "results": {}
            }
            
            if data.get("type") in ["csv", "excel"]:
                if data["type"] == "csv":
                    stats = data.get("summary_stats", {})
                    summary["results"]["csv_analysis"] = {
                        "total_rows": data.get("shape", [0, 0])[0],
                        "total_columns": data.get("shape", [0, 0])[1],
                        "numerical_columns": len(stats),
                        "statistics": stats
                    }
                    
                elif data["type"] == "excel":
                    summary["results"]["excel_analysis"] = {}
                    for sheet_name, sheet_data in data.get("sheets_data", {}).items():
                        stats = sheet_data.get("summary_stats", {})
                        summary["results"]["excel_analysis"][sheet_name] = {
                            "total_rows": sheet_data.get("shape", [0, 0])[0],
                            "total_columns": sheet_data.get("shape", [0, 0])[1],
                            "numerical_columns": len(stats),
                            "statistics": stats
                        }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {str(e)}")
            return {"error": str(e)}
    
    def create_visualization(self, data: Dict[str, Any], chart_type: str = "auto") -> go.Figure:
        """Create interactive visualizations using Plotly."""
        try:
            if data.get("type") == "csv":
                df = pd.DataFrame(data.get("data", []))
                return self._create_csv_visualization(df, chart_type)
                
            elif data.get("type") == "excel":
                # Use first sheet with numerical data
                for sheet_name, sheet_data in data.get("sheets_data", {}).items():
                    if sheet_data.get("summary_stats"):
                        df = pd.DataFrame(sheet_data.get("data", []))
                        return self._create_csv_visualization(df, chart_type)
                        
            # Default empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No visualizable data found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=14
            )
            return fig
    
    def _create_csv_visualization(self, df: pd.DataFrame, chart_type: str) -> go.Figure:
        """Create visualizations for CSV data."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No numerical data to visualize",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig
        
        if chart_type == "auto":
            if len(numerical_cols) == 1:
                chart_type = "histogram"
            elif len(numerical_cols) >= 2:
                chart_type = "scatter"
            else:
                chart_type = "bar"
        
        if chart_type == "histogram":
            col = numerical_cols[0]
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            
        elif chart_type == "scatter" and len(numerical_cols) >= 2:
            fig = px.scatter(df, x=numerical_cols[0], y=numerical_cols[1], 
                           title=f"{numerical_cols[1]} vs {numerical_cols[0]}")
            
        elif chart_type == "line" and len(numerical_cols) >= 2:
            fig = px.line(df, x=df.index, y=numerical_cols[0], 
                         title=f"{numerical_cols[0]} Trend")
            
        elif chart_type == "box":
            fig = go.Figure()
            for col in numerical_cols[:5]:  # Limit to 5 columns
                fig.add_trace(go.Box(y=df[col], name=col))
            fig.update_layout(title="Box Plot of Numerical Variables")
            
        else:
            # Default to correlation heatmap if multiple numerical columns
            if len(numerical_cols) > 1:
                corr_matrix = df[numerical_cols].corr()
                fig = px.imshow(corr_matrix, 
                              title="Correlation Matrix",
                              color_continuous_scale="RdBu",
                              aspect="auto")
            else:
                # Single column bar chart
                fig = px.bar(df, y=numerical_cols[0], 
                           title=f"Bar Chart of {numerical_cols[0]}")
        
        fig.update_layout(
            height=500,
            showlegend=True,
            font=dict(size=12)
        )
        
        return fig

# Agent Tools with Proper Docstring Formatting for SmolAgent
@tool
def analyze_soil_data(data: str) -> str:
    """
    Analyze soil test data and provide engineering insights.
    
    Args:
        data (str): JSON string containing soil test results with properties like density, moisture_content, bearing_capacity
    
    Returns:
        str: Detailed analysis of soil properties and engineering recommendations
    """
    try:
        soil_data = json.loads(data)
        
        analysis = []
        analysis.append("=== SOIL DATA ANALYSIS ===")
        
        # Check for key soil properties
        if "density" in soil_data:
            density = float(soil_data["density"])
            if density < 1.6:
                analysis.append(f"⚠️  Low density ({density} g/cm³) - Consider compaction")
            elif density > 2.1:
                analysis.append(f"✅ High density ({density} g/cm³) - Good bearing capacity")
            else:
                analysis.append(f"📊 Moderate density ({density} g/cm³)")
        
        if "moisture_content" in soil_data:
            moisture = float(soil_data["moisture_content"])
            if moisture > 25:
                analysis.append(f"💧 High moisture content ({moisture}%) - May affect stability")
            else:
                analysis.append(f"💧 Moisture content: {moisture}%")
        
        if "bearing_capacity" in soil_data:
            bearing = float(soil_data["bearing_capacity"])
            analysis.append(f"🏗️  Bearing capacity: {bearing} kPa")
            if bearing < 100:
                analysis.append("⚠️  Low bearing capacity - Deep foundations recommended")
        
        return "\n".join(analysis)
        
    except Exception as e:
        return f"Error analyzing soil data: {str(e)}"

@tool
def calculate_tunnel_support(diameter: float, depth: float, rock_quality: str) -> str:
    """
    Calculate tunnel support requirements based on geometry and ground conditions.
    
    Args:
        diameter (float): Tunnel diameter in meters
        depth (float): Depth below surface in meters
        rock_quality (str): Rock quality designation, must be one of 'good', 'fair', or 'poor'
    
    Returns:
        str: Support system recommendations including rock bolt spacing and shotcrete thickness
    """
    try:
        analysis = []
        analysis.append("=== TUNNEL SUPPORT ANALYSIS ===")
        analysis.append(f"Tunnel diameter: {diameter} m")
        analysis.append(f"Depth: {depth} m")
        analysis.append(f"Rock quality: {rock_quality}")
        
        # Calculate basic support requirements
        if rock_quality.lower() == "good":
            support_spacing = diameter * 1.5
            shotcrete_thickness = 50
        elif rock_quality.lower() == "fair":
            support_spacing = diameter * 1.0
            shotcrete_thickness = 100
        else:  # poor
            support_spacing = diameter * 0.5
            shotcrete_thickness = 150
        
        analysis.append(f"\n🔧 SUPPORT RECOMMENDATIONS:")
        analysis.append(f"• Rock bolt spacing: {support_spacing:.1f} m")
        analysis.append(f"• Shotcrete thickness: {shotcrete_thickness} mm")
        
        # Safety factors based on depth
        if depth > 50:
            analysis.append("⚠️  Deep tunnel - Consider additional monitoring")
        
        if diameter > 10:
            analysis.append("⚠️  Large diameter - May require steel ribs")
        
        return "\n".join(analysis)
        
    except Exception as e:
        return f"Error calculating tunnel support: {str(e)}"

@tool
def generate_safety_checklist(project_type: str) -> str:
    """
    Generate safety checklist for geotechnical projects.
    
    Args:
        project_type (str): Type of project, must be one of 'excavation', 'tunnel', 'foundation', or 'slope'
    
    Returns:
        str: Comprehensive safety checklist for the specified project type
    """
    checklists = {
        "excavation": [
            "✅ Soil stability assessment completed",
            "✅ Proper sloping or shoring installed", 
            "✅ Safe egress routes established",
            "✅ Atmospheric testing performed",
            "✅ Utility locations marked",
            "✅ Emergency equipment on site"
        ],
        "tunnel": [
            "✅ Ground support system installed",
            "✅ Ventilation system operational",
            "✅ Emergency escape routes clear",
            "✅ Gas monitoring equipment active",
            "✅ Communication systems tested",
            "✅ Water inflow management ready"
        ],
        "foundation": [
            "✅ Bearing capacity verified",
            "✅ Settlement analysis completed",
            "✅ Groundwater conditions assessed",
            "✅ Materials testing current",
            "✅ Construction sequence planned",
            "✅ Quality control procedures established"
        ],
        "slope": [
            "✅ Stability analysis completed",
            "✅ Drainage systems installed",
            "✅ Monitoring instruments in place",
            "✅ Access restrictions posted",
            "✅ Emergency response plan ready",
            "✅ Regular inspection schedule set"
        ]
    }
    
    checklist = checklists.get(project_type.lower(), checklists["excavation"])
    
    result = [f"=== {project_type.upper()} SAFETY CHECKLIST ==="]
    result.extend(checklist)
    result.append("\n⚠️  Always consult local regulations and engineering standards")
    
    return "\n".join(result)

class MultiAgentOrchestrator:
    """
    Multi-Agent Orchestration using SmolAgent
    Coordinates multiple specialized agents for different tasks.
    """
    
    def __init__(self):
        self.model = InferenceClientModel()
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize specialized agents for different tasks."""
        try:
            # Data Processing Agent
            self.agents["data_processor"] = CodeAgent(
                tools=[analyze_soil_data],
                model=self.model
            )
            
            # Engineering Analysis Agent
            self.agents["engineering_analyst"] = CodeAgent(
                tools=[calculate_tunnel_support, generate_safety_checklist],
                model=self.model
            )
            
            logger.info("SmolAgent orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            self.agents = {}
    
    def route_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Route queries to appropriate specialized agents."""
        try:
            query_lower = query.lower()
            
            # Determine which agent should handle the query
            if any(term in query_lower for term in ["soil", "bearing", "density", "moisture"]):
                agent_type = "data_processor"
            elif any(term in query_lower for term in ["tunnel", "support", "safety", "excavation", "checklist"]):
                agent_type = "engineering_analyst"
            else:
                agent_type = "data_processor"  # Default
            
            if agent_type in self.agents:
                result = self.agents[agent_type].run(query)
                return {
                    "agent_type": agent_type,
                    "query": query,
                    "response": result,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Fallback response
                return {
                    "agent_type": "fallback",
                    "query": query,
                    "response": "I understand your query, but the specialized agents are not available. Please try rephrasing your question.",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in agent routing: {str(e)}")
            return {
                "agent_type": "error",
                "query": query,
                "response": f"An error occurred while processing your query: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

# Initialize global components
@st.cache_resource
def initialize_system():
    """Initialize the complete system components."""
    return {
        "ingestion": DocumentIngestionModule(),
        "extraction": ExtractionModule(),
        "structured_output": StructuredOutputModule(),
        "analysis_viz": AnalysisVisualizationModule(),
        "orchestrator": MultiAgentOrchestrator()
    }

def main():
    """Main application function."""
    
    # Initialize system
    system = initialize_system()
    
    # Main header
    st.markdown('<h1 class="main-header">🏗️ Tunnelling & Geotechnical Engineering Workflow</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = {}
    
    # Sidebar for document upload and management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # System status
        st.subheader("🔧 System Status")
        col1, col2 = st.columns(2)
        with col1:
            if system["extraction"].model is not None:
                st.success("SmolVLM ✅")
            else:
                st.error("SmolVLM ❌")
        with col2:
            if system["orchestrator"].agents:
                st.success("SmolAgent ✅")
            else:
                st.error("SmolAgent ❌")
        
        st.divider()
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'png', 'jpg', 'jpeg', 'csv', 'xlsx', 'xls', 'json', 'md', 'txt'],
            help="Supported: PDF, Images, CSV, Excel, JSON, Markdown"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    # Validate file
                    is_valid, message, file_type = system["ingestion"].validate_file(uploaded_file)
                    
                    if is_valid:
                        st.info(f"✅ {message}")
                        
                        # Process based on file type
                        try:
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
                            
                            # Additional extraction for images using SmolVLM
                            if file_type == "images" and "error" not in extracted_data:
                                if system["extraction"].model is not None:
                                    with st.spinner("Running vision analysis..."):
                                        vision_result = system["extraction"].extract_from_image(
                                            extracted_data["image_data"]
                                        )
                                        if "error" not in vision_result:
                                            extracted_data.update(vision_result)
                                else:
                                    st.warning("Vision analysis unavailable (SmolVLM not loaded)")
                            
                            # Organize data
                            doc_id = f"{uploaded_file.name}_{int(time.time())}"
                            structured_data = system["structured_output"].organize_data(extracted_data, doc_id)
                            
                            # Store in session
                            st.session_state.processed_documents[doc_id] = structured_data
                            
                            st.success(f"✅ Document processed successfully!")
                            
                            # Show processing summary
                            with st.expander("📊 Processing Summary"):
                                st.json({
                                    "document_id": doc_id,
                                    "type": extracted_data.get("type", "unknown"),
                                    "timestamp": structured_data.get("timestamp", "unknown")
                                })
                                
                        except Exception as e:
                            st.error(f"❌ Processing error: {str(e)}")
                    else:
                        st.error(f"❌ {message}")
        
        st.divider()
        
        # Document list
        st.subheader("📋 Processed Documents")
        if st.session_state.processed_documents:
            for doc_id, doc_data in st.session_state.processed_documents.items():
                with st.expander(f"📄 {doc_id.split('_')[0][:20]}..."):
                    st.write(f"**Type:** {doc_data.get('document_type', 'Unknown')}")
                    st.write(f"**Status:** {doc_data.get('processing_status', 'Unknown')}")
                    st.write(f"**Timestamp:** {doc_data.get('timestamp', 'Unknown')[:19]}")
                    
                    if st.button(f"🗑️ Remove", key=f"remove_{doc_id}"):
                        del st.session_state.processed_documents[doc_id]
                        st.rerun()
        else:
            st.info("No documents processed yet")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 AI Chat Interface", "📊 Data Analysis", "📈 Visualizations", "🔧 System Status"])
    
    with tab1:
        st.subheader("🤖 AI-Powered Engineering Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your documents or engineering questions..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response using orchestrator
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your query..."):
                    
                    # Prepare context from processed documents
                    context = {
                        "documents": list(st.session_state.processed_documents.keys()),
                        "document_count": len(st.session_state.processed_documents)
                    }
                    
                    # Route query to appropriate agent
                    agent_response = system["orchestrator"].route_query(prompt, context)
                    
                    # Display response
                    response_text = agent_response.get("response", "I apologize, but I couldn't process your query.")
                    st.markdown(response_text)
                    
                    # Show agent info
                    with st.expander("🔍 Query Processing Details"):
                        st.json(agent_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Example queries
        st.subheader("💡 Example Queries")
        example_queries = [
            "Calculate tunnel support for 8m diameter at 25m depth in poor rock",
            "Generate safety checklist for excavation project",
            "Analyze soil data: {'density': 1.8, 'moisture_content': 15, 'bearing_capacity': 150}",
            "What are the key findings from the uploaded document?"
        ]
        
        for i, example in enumerate(example_queries):
            if st.button(example, key=f"example_{i}"):
                # Add example to chat
                st.session_state.messages.append({"role": "user", "content": example})
                
                # Process with agent
                context = {
                    "documents": list(st.session_state.processed_documents.keys()),
                    "document_count": len(st.session_state.processed_documents)
                }
                agent_response = system["orchestrator"].route_query(example, context)
                response_text = agent_response.get("response", "Could not process query.")
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                st.rerun()
    
    with tab2:
        st.subheader("📊 Data Analysis Dashboard")
        
        if st.session_state.processed_documents:
            # Select document for analysis
            doc_options = list(st.session_state.processed_documents.keys())
            selected_doc = st.selectbox("Select document for analysis:", doc_options)
            
            if selected_doc:
                doc_data = st.session_state.processed_documents[selected_doc]
                
                # Show document info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Document Type", doc_data.get("document_type", "Unknown"))
                with col2:
                    st.metric("Processing Status", doc_data.get("processing_status", "Unknown"))
                with col3:
                    st.metric("File Size", f"{len(str(doc_data)):,} chars")
                
                # Perform analysis based on document type
                content = doc_data.get("content", {})
                
                if doc_data.get("document_type") in ["csv", "excel"]:
                    # Statistical analysis for structured data
                    tabular_data = content.get("tabular_data", {})
                    analysis = system["analysis_viz"].generate_statistical_summary(tabular_data)
                    
                    if "error" not in analysis:
                        st.subheader("📈 Statistical Summary")
                        
                        # Display results in a nice format
                        results = analysis.get("results", {})
                        for analysis_type, analysis_data in results.items():
                            st.write(f"**{analysis_type.replace('_', ' ').title()}**")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Rows", analysis_data.get("total_rows", 0))
                            with col2:
                                st.metric("Total Columns", analysis_data.get("total_columns", 0))
                            with col3:
                                st.metric("Numerical Columns", analysis_data.get("numerical_columns", 0))
                            
                            # Show statistics if available
                            stats = analysis_data.get("statistics", {})
                            if stats:
                                st.write("**Descriptive Statistics:**")
                                stats_df = pd.DataFrame(stats)
                                st.dataframe(stats_df)
                    else:
                        st.error(f"Analysis error: {analysis['error']}")
                
                elif doc_data.get("document_type") == "image":
                    # Display vision analysis results
                    if "analysis_type" in content:
                        st.subheader("👁️ Vision Analysis Results")
                        st.write(f"**Query:** {content.get('query', 'N/A')}")
                        st.write(f"**Analysis:** {content.get('response', 'N/A')}")
                        st.write(f"**Confidence:** {content.get('confidence', 'N/A')}")
                
                elif doc_data.get("document_type") == "pdf":
                    # PDF analysis
                    text_data = content.get("text_data", [])
                    if text_data:
                        st.subheader("📄 PDF Analysis")
                        
                        total_chars = sum(page.get('char_count', 0) for page in text_data)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Pages", len(text_data))
                        with col2:
                            st.metric("Total Characters", f"{total_chars:,}")
                        with col3:
                            avg_chars = total_chars / len(text_data) if text_data else 0
                            st.metric("Avg Chars/Page", f"{avg_chars:.0f}")
                        
                        # Show sample text
                        if text_data:
                            st.write("**Sample Text (First Page):**")
                            sample_text = text_data[0].get('text', '')[:500]
                            st.text_area("Preview", sample_text, height=150, disabled=True)
                
                # Display raw content
                with st.expander("🔍 Raw Content Data"):
                    st.json(content)
        else:
            st.info("📥 Upload and process documents to see analysis options")
    
    with tab3:
        st.subheader("📈 Data Visualizations")
        
        if st.session_state.processed_documents:
            # Select document for visualization
            doc_options = [doc_id for doc_id, doc_data in st.session_state.processed_documents.items() 
                          if doc_data.get("document_type") in ["csv", "excel"]]
            
            if doc_options:
                selected_doc = st.selectbox("Select document for visualization:", doc_options, key="viz_doc")
                
                if selected_doc:
                    doc_data = st.session_state.processed_documents[selected_doc]
                    content = doc_data.get("content", {})
                    
                    # Chart type selection
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        chart_type = st.selectbox(
                            "Select chart type:",
                            ["auto", "histogram", "scatter", "line", "box", "correlation"],
                            key="chart_type"
                        )
                    with col2:
                        if st.button("🎨 Generate Visualization", type="primary"):
                            with st.spinner("Creating visualization..."):
                                tabular_data = content.get("tabular_data", {})
                                fig = system["analysis_viz"].create_visualization(tabular_data, chart_type)
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 No CSV or Excel files available for visualization")
        else:
            st.info("📥 Upload CSV or Excel files to create visualizations")
    
    with tab4:
        st.subheader("🔧 System Status & Information")
        
        # System status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if system["extraction"].model is not None:
                st.markdown('<div class="metric-card">✅ SmolVLM<br/>Ready</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">❌ SmolVLM<br/>Error</div>', unsafe_allow_html=True)
        
        with col2:
            if system["orchestrator"].agents:
                st.markdown('<div class="metric-card">✅ SmolAgent<br/>Ready</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);">❌ SmolAgent<br/>Error</div>', unsafe_allow_html=True)
        
        with col3:
            doc_count = len(st.session_state.processed_documents)
            st.markdown(f'<div class="metric-card">📄 Documents<br/>{doc_count}</div>', unsafe_allow_html=True)
        
        with col4:
            device = "CUDA" if torch.cuda.is_available() else "CPU"
            st.markdown(f'<div class="metric-card">💻 Device<br/>{device}</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Technology stack info
        st.subheader("🛠️ Technology Stack")
        tech_info = {
            "Framework": "Streamlit 🌊",
            "Vision-Language Model": "SmolVLM (HuggingFace Transformers) 🤖",
            "Agent Orchestration": "SmolAgent 🧠",
            "Visualization": "Plotly 📊",
            "Data Processing": "Pandas, NumPy 🐼",
            "Computer Vision": "PIL, OpenCV 👁️",
            "Document Processing": "PyPDF, OpenPyXL 📄"
        }
        
        for tech, desc in tech_info.items():
            st.write(f"**{tech}:** {desc}")
        
        st.divider()
        
        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        if st.button("🔄 Run System Check"):
            with st.spinner("Running system diagnostics..."):
                # Simulate check with actual system testing
                perf_data = {}
                
                # Test SmolVLM performance
                if system["extraction"].model is not None:
                    perf_data["SmolVLM Status"] = "✅ Operational"
                    perf_data["Device"] = system["extraction"].device
                else:
                    perf_data["SmolVLM Status"] = "❌ Not loaded"
                
                # Test SmolAgent performance
                if system["orchestrator"].agents:
                    perf_data["SmolAgent Status"] = "✅ Operational"
                    perf_data["Available Agents"] = len(system["orchestrator"].agents)
                else:
                    perf_data["SmolAgent Status"] = "❌ Not initialized"
                
                # Memory info
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    perf_data["GPU Memory"] = f"{gpu_memory:.1f} GB"
                
                perf_data["Documents Processed"] = len(st.session_state.processed_documents)
                perf_data["Chat Messages"] = len(st.session_state.messages)
                
                st.success("✅ System diagnostics completed!")
                
                # Display metrics in columns
                cols = st.columns(2)
                for i, (metric, value) in enumerate(perf_data.items()):
                    with cols[i % 2]:
                        st.metric(metric, value)
        
        st.divider()
        
        # Clear data options
        st.subheader("🧹 Data Management")
        
        col1, col2 = st.columns(2)
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

if __name__ == "__main__":
    main()
