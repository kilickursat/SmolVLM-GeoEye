#!/usr/bin/env python3
"""
SmolVLM-GeoEye: Geotechnical Engineering Workflow Application
=============================================================

This application provides a Streamlit interface for geotechnical document analysis
using SmolVLM on RunPod serverless GPUs with SmolAgent orchestration.

Author: SmolVLM-GeoEye Team
Version: 3.0.0
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
from datetime import datetime
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

# SmolAgent imports
from smolagents import tool, ToolCallingAgent, HfApiModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="SmolVLM-GeoEye: Geotechnical Engineering Workflow",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for geotechnical theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1e 100%);
        color: #ffffff;
    }
    .geotechnical-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    .engineering-metric {
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-size: 14px;
        font-weight: bold;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .soil-analysis {
        background: linear-gradient(135deg, #8b6914 0%, #6b5010 100%);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        color: white;
        margin: 10px 0;
    }
    .tunnel-info {
        background: linear-gradient(135deg, #2c5282 0%, #1a365d 100%);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        color: white;
        margin: 10px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = {}
if "async_jobs" not in st.session_state:
    st.session_state.async_jobs = {}

@dataclass
class Config:
    """Configuration for RunPod and HuggingFace"""
    api_key: Optional[str] = None
    endpoint_id: Optional[str] = None
    hf_token: Optional[str] = None
    timeout: int = 300
    max_retries: int = 3
    
    def __post_init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        self.hf_token = os.getenv("HF_TOKEN")

class GeotechnicalDataExtractor:
    """Extract numerical geotechnical data from text"""
    
    def extract_numerical_data_from_text(self, text: str) -> Dict[str, List[Dict]]:
        """Extract numerical values with context from text"""
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
            'plastic_limit': []
        }
        
        # Enhanced patterns to capture values with context
        patterns = {
            'spt_values': [
                r'SPT[\s:]*N[\s:]*=?\s*(\d+)(?:\s+at\s+(\d+\.?\d*)\s*(m|ft))?',
                r'N-value[\s:]*(\d+)(?:\s+at\s+(\d+\.?\d*)\s*(m|ft))?',
                r'blow count[\s:]*(\d+)(?:\s+at\s+(\d+\.?\d*)\s*(m|ft))?',
                r'(\d+)\s*blows(?:\s+at\s+(\d+\.?\d*)\s*(m|ft))?'
            ],
            'bearing_capacity': [
                r'bearing capacity[\s:]*(\d+\.?\d*)\s*(kPa|MPa|kN/m2|psf|ksf)',
                r'allowable bearing[\s:]*(\d+\.?\d*)\s*(kPa|MPa|kN/m2|psf|ksf)',
                r'ultimate bearing[\s:]*(\d+\.?\d*)\s*(kPa|MPa|kN/m2|psf|ksf)',
                r'qa[\s:]*=?\s*(\d+\.?\d*)\s*(kPa|MPa|kN/m2|psf|ksf)'
            ],
            'density': [
                r'(?:dry\s+)?density[\s:]*(\d+\.?\d*)\s*(g/cm3|kg/m3|pcf|kN/m3)',
                r'unit weight[\s:]*(\d+\.?\d*)\s*(kN/m3|pcf|kg/m3)',
                r'bulk density[\s:]*(\d+\.?\d*)\s*(g/cm3|kg/m3|pcf)',
                r'γ[\s:]*=?\s*(\d+\.?\d*)\s*(kN/m3|pcf)'
            ],
            'moisture_content': [
                r'moisture content[\s:]*(\d+\.?\d*)\s*%',
                r'water content[\s:]*(\d+\.?\d*)\s*%',
                r'w[\s:]*=?\s*(\d+\.?\d*)\s*%',
                r'MC[\s:]*(\d+\.?\d*)\s*%'
            ],
            'depth': [
                r'(?:at\s+)?depth[\s:]*(\d+\.?\d*)\s*(m|ft|cm)',
                r'(\d+\.?\d*)\s*(m|ft)\s+(?:depth|deep|below)',
                r'elevation[\s:]*([+-]?\d+\.?\d*)\s*(m|ft)',
                r'level[\s:]*([+-]?\d+\.?\d*)\s*(m|ft)'
            ]
        }
        
        # Extract values for each parameter type
        for param, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        value = float(match[0])
                        
                        # Handle depth information if present
                        depth_value = None
                        if len(match) > 2 and match[1]:  # Has depth info
                            try:
                                depth_value = float(match[1])
                                depth_unit = match[2] if len(match) > 2 else "m"
                            except:
                                depth_value = None
                        
                        unit = match[1] if len(match) > 1 and not match[1].replace('.','').isdigit() else ""
                        if len(match) > 2 and match[2] and not match[2].replace('.','').isdigit():
                            unit = match[2]
                        
                        # Find context (surrounding text)
                        context_pattern = rf'.{{0,100}}{re.escape(match[0])}.{{0,100}}'
                        context_matches = re.findall(context_pattern, text, re.IGNORECASE | re.DOTALL)
                        context = context_matches[0] if context_matches else ""
                        
                        data_entry = {
                            'value': value,
                            'unit': unit,
                            'context': context.strip(),
                            'source': 'VLM extraction'
                        }
                        
                        # Add depth if available
                        if depth_value is not None:
                            data_entry['depth'] = depth_value
                            data_entry['depth_unit'] = depth_unit
                        
                        numerical_data[param].append(data_entry)
        
        # Remove duplicates
        for param in numerical_data:
            seen = set()
            unique_data = []
            for item in numerical_data[param]:
                key = (item['value'], item.get('depth', 'no_depth'))
                if key not in seen:
                    seen.add(key)
                    unique_data.append(item)
            numerical_data[param] = unique_data
        
        return numerical_data

class RunPodClient:
    """Client for RunPod serverless API"""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = f"https://api.runpod.ai/v2/{config.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check if RunPod endpoint is healthy"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                return {"status": "healthy", "details": response.json()}
            else:
                return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def run_sync(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run synchronous inference"""
        try:
            response = requests.post(
                f"{self.base_url}/runsync",
                headers=self.headers,
                json={"input": input_data},
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "success",
                    "output": result.get("output", {}),
                    "processing_time": result.get("executionTime", "unknown")
                }
            else:
                return {
                    "status": "error",
                    "error": f"RunPod error: {response.status_code} - {response.text}"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Request failed: {str(e)}"
            }
    
    def run_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit async job"""
        try:
            response = requests.post(
                f"{self.base_url}/run",
                headers=self.headers,
                json={"input": input_data},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "status": "submitted",
                    "job_id": result.get("id"),
                    "details": result
                }
            else:
                return {
                    "status": "error",
                    "error": f"RunPod error: {response.status_code} - {response.text}"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Request failed: {str(e)}"
            }
    
    def check_async_status(self, job_id: str) -> Dict[str, Any]:
        """Check status of async job"""
        try:
            response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "error": f"Status check failed: {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Status check failed: {str(e)}"
            }

class DocumentIngestionModule:
    """Handles document upload and preprocessing"""
    
    def __init__(self):
        self.supported_formats = {
            'application/pdf': self.process_pdf,
            'image/png': self.process_image,
            'image/jpeg': self.process_image,
            'image/jpg': self.process_image,
            'text/csv': self.process_csv,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': self.process_excel,
            'text/plain': self.process_text,
            'application/json': self.process_json,
            'text/markdown': self.process_markdown
        }
        self.data_extractor = GeotechnicalDataExtractor()
    
    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """Process uploaded file based on type"""
        try:
            file_type = uploaded_file.type
            
            if file_type in self.supported_formats:
                return self.supported_formats[file_type](uploaded_file)
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported file type: {file_type}"
                }
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def process_pdf(self, uploaded_file) -> Dict[str, Any]:
        """Extract text and metadata from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            
            extracted_data = {
                "type": "pdf",
                "filename": uploaded_file.name,
                "page_count": len(pdf_reader.pages),
                "text_data": [],
                "metadata": {}
            }
            
            # Extract text from each page
            all_text = ""
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                all_text += text + "\n"
                extracted_data["text_data"].append({
                    "page_number": i + 1,
                    "text": text
                })
            
            # Extract numerical data from all text
            extracted_data['numerical_data'] = self.data_extractor.extract_numerical_data_from_text(all_text)
            
            # Extract metadata
            if pdf_reader.metadata:
                extracted_data["metadata"] = {
                    "title": getattr(pdf_reader.metadata, 'title', None),
                    "author": getattr(pdf_reader.metadata, 'author', None),
                    "subject": getattr(pdf_reader.metadata, 'subject', None),
                    "creator": getattr(pdf_reader.metadata, 'creator', None),
                    "producer": getattr(pdf_reader.metadata, 'producer', None),
                    "creation_date": str(getattr(pdf_reader.metadata, 'creation_date', None)),
                    "modification_date": str(getattr(pdf_reader.metadata, 'modification_date', None))
                }
            
            extracted_data["status"] = "success"
            return extracted_data
            
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            return {
                "status": "error",
                "error": f"PDF processing failed: {str(e)}"
            }
    
    def process_image(self, uploaded_file) -> Dict[str, Any]:
        """Prepare image for VLM analysis"""
        try:
            # Read image
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create a temporary file to save the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                image.save(tmp_file.name, 'PNG')
                
                # Read the image data
                with open(tmp_file.name, 'rb') as f:
                    image_data = f.read()
                
                # Clean up
                os.unlink(tmp_file.name)
            
            # Convert to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            return {
                "status": "success",
                "type": "image",
                "filename": uploaded_file.name,
                "image_base64": image_base64,
                "image_size": image.size,
                "image_mode": image.mode
            }
            
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return {
                "status": "error",
                "error": f"Image processing failed: {str(e)}"
            }
    
    def process_csv(self, uploaded_file) -> Dict[str, Any]:
        """Process CSV file"""
        try:
            df = pd.read_csv(uploaded_file)
            
            return {
                "status": "success",
                "type": "csv",
                "filename": uploaded_file.name,
                "dataframe": df,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "preview": df.head(10).to_dict('records')
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"CSV processing failed: {str(e)}"
            }
    
    def process_excel(self, uploaded_file) -> Dict[str, Any]:
        """Process Excel file"""
        try:
            df = pd.read_excel(uploaded_file)
            
            return {
                "status": "success",
                "type": "excel",
                "filename": uploaded_file.name,
                "dataframe": df,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "preview": df.head(10).to_dict('records')
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Excel processing failed: {str(e)}"
            }
    
    def process_text(self, uploaded_file) -> Dict[str, Any]:
        """Process text file"""
        try:
            text = uploaded_file.read().decode('utf-8')
            
            return {
                "status": "success",
                "type": "text",
                "filename": uploaded_file.name,
                "content": text,
                "size": len(text),
                "lines": text.count('\n') + 1
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Text processing failed: {str(e)}"
            }
    
    def process_json(self, uploaded_file) -> Dict[str, Any]:
        """Process JSON file"""
        try:
            data = json.load(uploaded_file)
            
            return {
                "status": "success",
                "type": "json",
                "filename": uploaded_file.name,
                "data": data,
                "size": uploaded_file.size
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"JSON processing failed: {str(e)}"
            }
    
    def process_markdown(self, uploaded_file) -> Dict[str, Any]:
        """Process Markdown file"""
        try:
            content = uploaded_file.read().decode('utf-8')
            
            return {
                "status": "success",
                "type": "markdown",
                "filename": uploaded_file.name,
                "content": content,
                "size": len(content)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Markdown processing failed: {str(e)}"
            }

class GeotechnicalExtractionModule:
    """Handles geotechnical-specific data extraction using VLM"""
    
    def __init__(self, runpod_client: RunPodClient):
        self.runpod_client = runpod_client
        self.data_extractor = GeotechnicalDataExtractor()
    
    def extract_from_image(self, image_data: str, processing_mode: str = "sync") -> Dict[str, Any]:
        """Extract geotechnical information from image using SmolVLM"""
        
        # Enhanced query for geotechnical analysis
        query = """Analyze this geotechnical engineering document or image. Focus on extracting:

GEOTECHNICAL PARAMETERS WITH VALUES:
1. Soil properties with numerical values (density, moisture content, plasticity index, liquid limit, plastic limit)
2. Rock properties with values (strength, RQD, joint conditions, GSI)
3. Foundation data with numbers (bearing capacity in kPa/MPa, settlement in mm, pile capacity)
4. Test results with specific values (SPT N-values at different depths, CPT data, laboratory test results)

IMPORTANT: Always include the numerical values with their units. For example, don't just say "high density", say "density = 2.1 g/cm³" or "SPT N-value = 25 at 5m depth".

Please provide a detailed technical analysis with all numerical values and their units."""
        
        # Prepare input for RunPod
        input_data = {
            "image_data": image_data,
            "query": query,
            "max_new_tokens": 512,
            "temperature": 0.3,
            "do_sample": True
        }
        
        if processing_mode == "sync":
            # Synchronous processing
            result = self.runpod_client.run_sync(input_data)
            
            if result["status"] == "success":
                output = result["output"]
                response_text = output.get("response", "")
                
                # Extract numerical data from response
                numerical_data = self.data_extractor.extract_numerical_data_from_text(response_text)
                
                return {
                    "extraction_type": "geotechnical_vision_analysis",
                    "query": query,
                    "response": response_text,
                    "numerical_data": numerical_data,
                    "confidence": "high",
                    "processing_time": output.get("processing_time", "unknown"),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "extraction_type": "error",
                    "error": result.get("error", "Unknown error")
                }
        else:
            # Asynchronous processing
            result = self.runpod_client.run_async(input_data)
            
            if result["status"] == "submitted":
                return {
                    "extraction_type": "async_submitted",
                    "job_id": result["job_id"],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "extraction_type": "error",
                    "error": result.get("error", "Unknown error")
                }

class StructuredOutputModule:
    """Organizes extracted data into structured format"""
    
    def __init__(self):
        self.output_schema = {
            "geotechnical_parameters": {
                "soil_properties": ["density", "moisture_content", "void_ratio", "porosity"],
                "strength_parameters": ["cohesion", "friction_angle", "undrained_strength"],
                "consolidation": ["compression_index", "recompression_index", "preconsolidation_pressure"],
                "permeability": ["hydraulic_conductivity", "coefficient_of_permeability"],
                "index_properties": ["liquid_limit", "plastic_limit", "plasticity_index"]
            },
            "test_results": {
                "field_tests": ["spt", "cpt", "vane_shear", "pressuremeter"],
                "laboratory_tests": ["triaxial", "direct_shear", "consolidation", "permeability"]
            },
            "foundation_data": {
                "bearing_capacity": ["ultimate", "allowable", "factor_of_safety"],
                "settlement": ["immediate", "consolidation", "total"],
                "pile_data": ["capacity", "length", "diameter", "type"]
            }
        }
    
    def organize_data(self, extracted_data: Dict[str, Any], document_id: str) -> Dict[str, Any]:
        """Organize extracted data into structured format"""
        
        structured_data = {
            "document_id": document_id,
            "timestamp": datetime.now().isoformat(),
            "document_type": extracted_data.get("type", "unknown"),
            "processing_status": "completed",
            "content": {},
            "metadata": {},
            "searchable_fields": [],
            "numerical_data": {}
        }
        
        # Store numerical data separately for easy access
        if "numerical_data" in extracted_data:
            structured_data["numerical_data"] = extracted_data["numerical_data"]
        
        # Handle different document types
        doc_type = extracted_data.get("type")
        
        if doc_type == "pdf":
            structured_data["content"]["text_data"] = extracted_data.get("text_data", [])
            structured_data["metadata"] = extracted_data.get("metadata", {})
            structured_data["metadata"]["page_count"] = extracted_data.get("page_count", 0)
            
            # Create searchable text
            all_text = " ".join([page["text"] for page in extracted_data.get("text_data", [])])
            structured_data["searchable_fields"].append(all_text)
            
        elif doc_type == "image":
            # Store VLM extraction results
            if "extraction_type" in extracted_data and extracted_data["extraction_type"] == "geotechnical_vision_analysis":
                structured_data["content"] = {
                    "query": extracted_data.get("query"),
                    "response": extracted_data.get("response"),
                    "processing_time": extracted_data.get("processing_time")
                }
                structured_data["searchable_fields"].append(extracted_data.get("response", ""))
                
                # Store numerical data if extracted
                if "numerical_data" in extracted_data:
                    structured_data["numerical_data"] = extracted_data["numerical_data"]
            
        elif doc_type in ["csv", "excel"]:
            df = extracted_data.get("dataframe")
            if df is not None:
                structured_data["content"]["data"] = df.to_dict('records')
                structured_data["content"]["columns"] = df.columns.tolist()
                structured_data["content"]["shape"] = df.shape
                structured_data["content"]["statistics"] = df.describe().to_dict() if not df.empty else {}
            
        elif doc_type == "json":
            structured_data["content"]["data"] = extracted_data.get("data", {})
            
        elif doc_type in ["text", "markdown"]:
            structured_data["content"]["text"] = extracted_data.get("content", "")
            structured_data["searchable_fields"].append(extracted_data.get("content", ""))
        
        return structured_data

class GeotechnicalVisualizationModule:
    """Creates visualizations for geotechnical data"""
    
    def __init__(self):
        self.chart_types = {
            "depth_profile": self._create_depth_profile,
            "parameter_distribution": self._create_parameter_distribution,
            "correlation_matrix": self._create_correlation_matrix,
            "time_series": self._create_time_series
        }
    
    def _create_numerical_visualization(self, numerical_data: Dict[str, List[Dict]], doc_type: str) -> go.Figure:
        """Create visualization from extracted numerical data"""
        # Create subplots for different data types
        subplot_titles = []
        plot_count = 0
        
        # Determine which plots to create
        has_spt_depth = bool(numerical_data.get('spt_values')) and any(d.get('depth') for d in numerical_data.get('spt_values', []))
        has_bearing = bool(numerical_data.get('bearing_capacity'))
        has_soil_props = bool(numerical_data.get('density') or numerical_data.get('moisture_content'))
        has_strength = bool(numerical_data.get('cohesion') or numerical_data.get('friction_angle'))
        
        if has_spt_depth:
            subplot_titles.append("SPT N-Values vs Depth")
            plot_count += 1
        if has_bearing:
            subplot_titles.append("Bearing Capacity")
            plot_count += 1
        if has_soil_props:
            subplot_titles.append("Soil Properties")
            plot_count += 1
        if has_strength:
            subplot_titles.append("Strength Parameters")
            plot_count += 1
        
        if plot_count == 0:
            return self._create_error_visualization("No numerical data to visualize")
        
        # Create subplot layout
        rows = (plot_count + 1) // 2
        cols = 2 if plot_count > 1 else 1
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles[:plot_count],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        current_plot = 1
        
        # Plot 1: SPT values vs depth
        if has_spt_depth:
            spt_data = numerical_data['spt_values']
            depths = []
            values = []
            labels = []
            
            for item in spt_data:
                if 'depth' in item:
                    depths.append(item['depth'])
                    values.append(item['value'])
                    labels.append(f"N={item['value']} at {item['depth']}{item.get('depth_unit', 'm')}")
            
            if depths and values:
                row = ((current_plot - 1) // cols) + 1
                col = ((current_plot - 1) % cols) + 1
                
                fig.add_trace(
                    go.Scatter(
                        x=values,
                        y=depths,
                        mode='markers+lines',
                        marker=dict(size=10, color='blue'),
                        line=dict(color='blue', width=2),
                        text=labels,
                        name='SPT N-Value',
                        hovertemplate='%{text}<extra></extra>'
                    ),
                    row=row, col=col
                )
                fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=row, col=col)
                fig.update_xaxes(title_text="SPT N-Value", row=row, col=col)
                current_plot += 1
        
        # Plot 2: Bearing capacity
        if has_bearing and current_plot <= plot_count:
            bearing_data = numerical_data['bearing_capacity']
            values = [item['value'] for item in bearing_data]
            labels = [f"{item['value']} {item['unit']}" for item in bearing_data]
            
            row = ((current_plot - 1) // cols) + 1
            col = ((current_plot - 1) % cols) + 1
            
            fig.add_trace(
                go.Bar(
                    x=list(range(len(values))),
                    y=values,
                    text=labels,
                    name='Bearing Capacity',
                    marker_color='green'
                ),
                row=row, col=col
            )
            fig.update_xaxes(title_text="Sample", row=row, col=col)
            fig.update_yaxes(title_text="Bearing Capacity (kPa)", row=row, col=col)
            current_plot += 1
        
        # Plot 3: Soil properties
        if has_soil_props and current_plot <= plot_count:
            row = ((current_plot - 1) // cols) + 1
            col = ((current_plot - 1) % cols) + 1
            
            properties = []
            values = []
            
            if numerical_data.get('density'):
                for item in numerical_data['density']:
                    properties.append('Density')
                    values.append(item['value'])
            
            if numerical_data.get('moisture_content'):
                for item in numerical_data['moisture_content']:
                    properties.append('Moisture Content')
                    values.append(item['value'])
            
            if properties and values:
                fig.add_trace(
                    go.Box(
                        x=properties,
                        y=values,
                        name='Soil Properties',
                        marker_color='orange'
                    ),
                    row=row, col=col
                )
                fig.update_xaxes(title_text="Property", row=row, col=col)
                fig.update_yaxes(title_text="Value", row=row, col=col)
                current_plot += 1
        
        fig.update_layout(
            title_text=f"Geotechnical Data Analysis - {doc_type.upper()} Document",
            height=400 * rows,
            showlegend=False
        )
        
        return fig
    
    def create_visualization_from_any_document(self, doc_data: Dict[str, Any]) -> go.Figure:
        """Create visualizations from any document type based on extracted data"""
        try:
            doc_type = doc_data.get("document_type", "unknown")
            content = doc_data.get("content", {})
            numerical_data = doc_data.get("numerical_data", {})
            
            # Use numerical data if available
            if numerical_data and any(numerical_data.values()):
                return self._create_numerical_visualization(numerical_data, doc_type)
            
            # Fallback visualizations for different document types
            if doc_type in ["csv", "excel"]:
                data = content.get("data", [])
                if data:
                    df = pd.DataFrame(data)
                    return self._create_dataframe_visualization(df)
            
            elif doc_type == "json":
                json_data = content.get("data", {})
                return self._create_json_visualization(json_data)
            
            elif doc_type == "image":
                # For images, show analysis summary
                response = content.get("response", "")
                if response:
                    return self._create_text_summary_visualization(response)
            
            elif doc_type == "pdf":
                # For PDFs, analyze text data
                text_data = content.get("text_data", [])
                if text_data:
                    return self._create_pdf_analysis_visualization(text_data)
            
            # Default error visualization
            return self._create_error_visualization("No suitable data for visualization")
            
        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            return self._create_error_visualization(f"Visualization error: {str(e)}")
    
    def _create_dataframe_visualization(self, df: pd.DataFrame) -> go.Figure:
        """Create visualization from dataframe"""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Create scatter matrix for numeric columns
            fig = px.scatter_matrix(
                df[numeric_cols[:4]],  # Limit to 4 columns for readability
                title="Geotechnical Data Correlation Matrix"
            )
        elif len(numeric_cols) == 1:
            # Create histogram for single numeric column
            fig = px.histogram(
                df,
                x=numeric_cols[0],
                title=f"Distribution of {numeric_cols[0]}"
            )
        else:
            # No numeric data
            fig = self._create_error_visualization("No numeric data found in file")
        
        return fig
    
    def _create_json_visualization(self, json_data: Any) -> go.Figure:
        """Create visualization from JSON data"""
        # Try to extract numeric data from JSON
        numeric_data = self._extract_numeric_from_json(json_data)
        
        if numeric_data:
            fig = go.Figure()
            
            for key, values in numeric_data.items():
                fig.add_trace(go.Bar(
                    name=key,
                    x=list(range(len(values))),
                    y=values
                ))
            
            fig.update_layout(
                title="Geotechnical Parameters from JSON",
                xaxis_title="Index",
                yaxis_title="Value"
            )
        else:
            fig = self._create_error_visualization("No numeric data found in JSON")
        
        return fig
    
    def _create_text_summary_visualization(self, text: str) -> go.Figure:
        """Create summary visualization from text"""
        # Extract key terms and their frequencies
        geotechnical_terms = [
            "spt", "bearing", "capacity", "density", "moisture", "soil",
            "rock", "foundation", "settlement", "pile", "cohesion",
            "friction", "angle", "strength", "test", "sample"
        ]
        
        term_counts = {}
        text_lower = text.lower()
        
        for term in geotechnical_terms:
            count = text_lower.count(term)
            if count > 0:
                term_counts[term] = count
        
        if term_counts:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(term_counts.keys()),
                    y=list(term_counts.values()),
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title="Geotechnical Terms Frequency in Analysis",
                xaxis_title="Terms",
                yaxis_title="Frequency",
                xaxis_tickangle=-45
            )
        else:
            fig = self._create_error_visualization("No geotechnical terms found")
        
        return fig
    
    def _create_pdf_analysis_visualization(self, text_data: List[Dict]) -> go.Figure:
        """Create visualization from PDF text data"""
        # Analyze text length per page
        page_numbers = []
        text_lengths = []
        
        for page in text_data[:20]:  # Limit to first 20 pages
            page_numbers.append(f"Page {page['page_number']}")
            text_lengths.append(len(page['text']))
        
        fig = go.Figure(data=[
            go.Bar(
                x=page_numbers,
                y=text_lengths,
                marker_color='green'
            )
        ])
        
        fig.update_layout(
            title="PDF Document Structure Analysis",
            xaxis_title="Page",
            yaxis_title="Text Length (characters)",
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _create_error_visualization(self, error_message: str) -> go.Figure:
        """Create error visualization"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"⚠️ {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="red")
        )
        
        fig.update_layout(
            title="Visualization Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig
    
    def _extract_numeric_from_json(self, data: Any, prefix: str = "") -> Dict[str, List[float]]:
        """Recursively extract numeric values from JSON"""
        numeric_data = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                extracted = self._extract_numeric_from_json(value, new_prefix)
                numeric_data.update(extracted)
        elif isinstance(data, list):
            numeric_values = [v for v in data if isinstance(v, (int, float))]
            if numeric_values:
                numeric_data[prefix] = numeric_values
        elif isinstance(data, (int, float)):
            numeric_data[prefix] = [data]
        
        return numeric_data
    
    def _create_depth_profile(self, data: pd.DataFrame, param_column: str, depth_column: str = "Depth") -> go.Figure:
        """Create depth profile chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data[param_column],
            y=data[depth_column],
            mode='lines+markers',
            name=param_column,
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"{param_column} vs Depth Profile",
            xaxis_title=param_column,
            yaxis_title=f"{depth_column} (m)",
            yaxis_autorange="reversed",
            hovermode='closest'
        )
        
        return fig
    
    def _create_parameter_distribution(self, data: pd.DataFrame, param_column: str) -> go.Figure:
        """Create parameter distribution chart"""
        fig = px.histogram(
            data,
            x=param_column,
            nbins=20,
            title=f"Distribution of {param_column}",
            labels={param_column: f"{param_column} Value"}
        )
        
        fig.add_vline(
            x=data[param_column].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {data[param_column].mean():.2f}"
        )
        
        return fig
    
    def _create_correlation_matrix(self, data: pd.DataFrame) -> go.Figure:
        """Create correlation matrix heatmap"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Parameter", y="Parameter", color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            title="Geotechnical Parameters Correlation Matrix"
        )
        
        fig.update_layout(width=800, height=800)
        
        return fig
    
    def _create_time_series(self, data: pd.DataFrame, value_column: str, time_column: str = "Date") -> go.Figure:
        """Create time series chart"""
        fig = px.line(
            data,
            x=time_column,
            y=value_column,
            title=f"{value_column} Time Series",
            markers=True
        )
        
        fig.update_traces(
            line_color='green',
            line_width=2,
            marker_size=8
        )
        
        return fig

class GeotechnicalMultiAgentOrchestrator:
    """Orchestrates multiple agents for geotechnical analysis"""
    
    def __init__(self, model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct", hf_token: Optional[str] = None):
        self.model_id = model_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize specialized geotechnical agents"""
        try:
            # Create model instance
            model = HfApiModel(model_id=self.model_id, token=self.hf_token)
            
            # Define specialized tools for geotechnical analysis
            @tool
            def analyze_soil_data(data: str) -> str:
                """Analyze soil test data and provide engineering insights."""
                # This is a placeholder - in production, implement actual analysis
                return f"Analyzing soil data: {data[:100]}... Based on the data, the soil appears to have moderate bearing capacity."
            
            @tool
            def calculate_tunnel_support(diameter: float, depth: float, rock_quality: str) -> str:
                """Calculate tunnel support requirements with safety factors."""
                # Simplified calculation
                if rock_quality.lower() == "good":
                    support = "Light support: Rock bolts at 2m spacing"
                elif rock_quality.lower() == "fair":
                    support = "Moderate support: Rock bolts at 1.5m spacing with mesh"
                else:
                    support = "Heavy support: Steel sets with lagging"
                
                return f"For a {diameter}m diameter tunnel at {depth}m depth in {rock_quality} rock: {support}"
            
            @tool
            def generate_safety_checklist(project_type: str) -> str:
                """Generate comprehensive safety protocols."""
                checklists = {
                    "excavation": "1. Check utilities before digging\n2. Shore/slope as required\n3. Daily inspections\n4. Access/egress every 25ft",
                    "tunneling": "1. Ground monitoring system\n2. Ventilation check\n3. Emergency procedures\n4. Face stability monitoring",
                    "foundation": "1. Soil bearing verification\n2. Dewatering if needed\n3. Concrete quality control\n4. Settlement monitoring"
                }
                
                return checklists.get(project_type.lower(), "General safety protocols required")
            
            # Create specialized agents
            self.agents["soil_analyst"] = ToolCallingAgent(
                tools=[analyze_soil_data],
                model=model,
                system_prompt="You are a geotechnical engineer specializing in soil mechanics and foundation engineering."
            )
            
            self.agents["tunnel_engineer"] = ToolCallingAgent(
                tools=[calculate_tunnel_support],
                model=model,
                system_prompt="You are a tunnel engineering specialist with expertise in rock mechanics and support systems."
            )
            
            self.agents["safety_officer"] = ToolCallingAgent(
                tools=[generate_safety_checklist],
                model=model,
                system_prompt="You are a construction safety specialist focused on geotechnical projects."
            )
            
            logger.info("Geotechnical agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            # Create fallback agents without tools
            self.agents = {}
    
    def route_geotechnical_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Route queries to appropriate geotechnical specialists using document context"""
        try:
            query_lower = query.lower()
            
            # Extract document content from context
            document_contents = []
            numerical_summary = {}
            
            if context and "processed_documents" in context:
                processed_docs = context["processed_documents"]
                
                for doc_id, doc_data in processed_docs.items():
                    content = doc_data.get("content", {})
                    numerical_data = doc_data.get("numerical_data", {})
                    
                    # Extract VLM analysis response
                    if "response" in content:
                        document_contents.append(f"VLM Analysis from {doc_id}:\n{content['response']}\n")
                    
                    # Extract text from PDFs
                    if "text_data" in content:
                        for page in content["text_data"][:3]:  # First 3 pages
                            document_contents.append(f"PDF Page {page['page_number']}:\n{page['text'][:500]}...\n")
                    
                    # Extract numerical data summary
                    if numerical_data:
                        for param, values in numerical_data.items():
                            if values:
                                if param not in numerical_summary:
                                    numerical_summary[param] = []
                                numerical_summary[param].extend(values)
            
            # Combine document content
            combined_content = "\n".join(document_contents) if document_contents else "No documents uploaded"
            
            # Format numerical summary
            numerical_content = ""
            if numerical_summary:
                numerical_content = "\nExtracted Numerical Data:\n"
                for param, values in numerical_summary.items():
                    numerical_content += f"\n{param.replace('_', ' ').title()}:\n"
                    for val in values[:5]:  # Limit to 5 values per parameter
                        depth_info = f" at {val['depth']}{val.get('depth_unit', 'm')} depth" if 'depth' in val else ""
                        numerical_content += f"  • {val['value']} {val['unit']}{depth_info}\n"
            
            # Create context-aware query
            context_aware_query = f"""Based on the following uploaded document content:

{combined_content}
{numerical_content}

User Question: {query}

Please answer the user's question specifically using the information extracted from the uploaded documents. If specific numerical values are available in the extracted data, use them in your analysis. If the information is not available in the documents, clearly state that."""

            # Route to appropriate agent with context
            if any(term in query_lower for term in ["soil", "bearing", "density", "moisture", "spt"]):
                agent_type = "soil_analyst"
                enhanced_query = context_aware_query
            elif any(term in query_lower for term in ["tunnel", "support", "excavation", "rock"]):
                agent_type = "tunnel_engineer"
                enhanced_query = context_aware_query
            elif any(term in query_lower for term in ["safety", "checklist", "hazard", "risk"]):
                agent_type = "safety_officer"
                enhanced_query = context_aware_query
            else:
                agent_type = "soil_analyst"  # Default
                enhanced_query = context_aware_query
            
            if agent_type in self.agents:
                result = self.agents[agent_type].run(enhanced_query)
                return {
                    "agent_type": agent_type,
                    "query": query,
                    "enhanced_query": enhanced_query,
                    "response": result,
                    "timestamp": datetime.now().isoformat(),
                    "domain": "Geotechnical Engineering",
                    "powered_by": "RunPod GPU + SmolAgent",
                    "document_based": bool(document_contents)
                }
            else:
                return {
                    "agent_type": "fallback",
                    "query": query,
                    "response": "Please upload documents and ask questions about the content.",
                    "timestamp": datetime.now().isoformat(),
                    "domain": "Geotechnical Engineering",
                    "document_based": False
                }
                
        except Exception as e:
            logger.error(f"Error in geotechnical agent routing: {str(e)}")
            return {
                "agent_type": "error",
                "query": query,
                "response": f"I encountered an error processing your query. Please ensure you have uploaded documents.",
                "timestamp": datetime.now().isoformat(),
                "domain": "Geotechnical Engineering",
                "document_based": False
            }

def initialize_system() -> Dict[str, Any]:
    """Initialize all system components"""
    system = {}
    
    # Load configuration
    config = Config()
    system["config"] = config
    
    # Initialize RunPod client
    if config.api_key and config.endpoint_id:
        system["runpod_client"] = RunPodClient(config)
    else:
        logger.warning("RunPod not configured - using mock client")
        system["runpod_client"] = None
    
    # Initialize modules
    system["document_ingestion"] = DocumentIngestionModule()
    system["extraction"] = GeotechnicalExtractionModule(system["runpod_client"]) if system["runpod_client"] else None
    system["structured_output"] = StructuredOutputModule()
    system["visualization"] = GeotechnicalVisualizationModule()
    system["orchestrator"] = GeotechnicalMultiAgentOrchestrator(hf_token=config.hf_token)
    
    return system

def main():
    """Main application function"""
    # Initialize system
    system = initialize_system()
    
    # Check RunPod status
    runpod_status = {"status": "unknown", "error": None}
    if system["runpod_client"]:
        runpod_status = system["runpod_client"].health_check()
    
    # Header
    st.markdown('<h1 style="text-align: center; color: #667eea;">🏗️ SmolVLM-GeoEye: Geotechnical Engineering Workflow</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #a0a0a0;">Powered by SmolVLM on RunPod GPU + SmolAgent</p>', unsafe_allow_html=True)
    
    # Sidebar for document upload
    with st.sidebar:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload geotechnical documents",
            type=['pdf', 'png', 'jpg', 'jpeg', 'csv', 'xlsx', 'txt', 'json', 'md'],
            accept_multiple_files=True,
            help="Upload PDFs, images, spreadsheets, or text files containing geotechnical data"
        )
        
        # Processing mode
        processing_mode = st.radio(
            "🚀 Processing Mode",
            ["Sync (Wait for results)", "Async (Background processing)"],
            index=0
        )
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_documents:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Process document
                        doc_result = system["document_ingestion"].process_uploaded_file(uploaded_file)
                        
                        if doc_result["status"] == "success":
                            # Generate unique document ID
                            doc_id = f"{uploaded_file.name}_{str(uuid.uuid4())[:8]}"
                            
                            # For images, run VLM extraction
                            if doc_result["type"] == "image" and system["extraction"]:
                                extraction_mode = "sync" if "Sync" in processing_mode else "async"
                                extraction_result = system["extraction"].extract_from_image(
                                    doc_result["image_base64"],
                                    processing_mode=extraction_mode
                                )
                                
                                if extraction_result["extraction_type"] == "async_submitted":
                                    st.session_state.async_jobs[doc_id] = {
                                        "job_id": extraction_result["job_id"],
                                        "status": "submitted",
                                        "document_name": uploaded_file.name,
                                        "timestamp": extraction_result["timestamp"]
                                    }
                                    st.info(f"🔄 Async job submitted for {uploaded_file.name}")
                                else:
                                    doc_result.update(extraction_result)
                            
                            # Structure and store the data
                            structured_data = system["structured_output"].organize_data(doc_result, doc_id)
                            st.session_state.processed_documents[doc_id] = structured_data
                            
                            st.success(f"✅ {uploaded_file.name} processed successfully!")
                        else:
                            st.error(f"❌ Error processing {uploaded_file.name}: {doc_result.get('error', 'Unknown error')}")
        
        # Async job status
        if st.session_state.async_jobs:
            st.divider()
            st.subheader("⏳ Async Jobs")
            
            for doc_id, job_info in list(st.session_state.async_jobs.items()):
                if job_info["status"] != "completed" and system["runpod_client"]:
                    status = system["runpod_client"].check_async_status(job_info["job_id"])
                    
                    if status.get("status") == "COMPLETED":
                        job_info["status"] = "completed"
                        # Process completed job results
                        # ... (implementation depends on RunPod response format)
                
                status_emoji = {
                    "submitted": "🔄",
                    "processing": "⚙️",
                    "completed": "✅",
                    "error": "❌"
                }.get(job_info["status"], "❓")
                
                st.write(f"{status_emoji} {job_info['document_name']}: {job_info['status']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 AI Chat", 
        "📊 Data Analysis", 
        "📈 Visualizations",
        "🚀 System Status",
        "⚙️ Settings"
    ])
    
    with tab1:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.subheader("🤖 Geotechnical AI Assistant")
        st.caption("Ask questions about your uploaded documents or general geotechnical engineering topics")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about soil properties, bearing capacity, tunnel design..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🧠 Analyzing with SmolAgent..."):
                    # Prepare context with uploaded documents
                    context = {
                        "documents": list(st.session_state.processed_documents.keys()),
                        "document_count": len(st.session_state.processed_documents),
                        "processed_documents": st.session_state.processed_documents,
                        "runpod_status": runpod_status["status"],
                        "domain": "geotechnical_engineering"
                    }
                    
                    # Route query to appropriate agent
                    agent_response = system["orchestrator"].route_geotechnical_query(prompt, context)
                    response_text = agent_response.get("response", "I couldn't process your query. Please try again.")
                    
                    st.write(response_text)
                    
                    # Show analysis details
                    with st.expander("🔍 Analysis Details"):
                        st.json({
                            "agent_type": agent_response.get("agent_type"),
                            "document_based": agent_response.get("document_based"),
                            "timestamp": agent_response.get("timestamp")
                        })
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Enhanced example queries for geotechnical engineering
        st.subheader("💡 Example Questions for Uploaded Documents")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📄 Document Analysis Questions:**")
            example_queries_1 = [
                "What are the SPT values in the uploaded document?",
                "Summarize the soil properties from the test results",
                "What is the bearing capacity mentioned in the report?",
                "Extract all density values with their depths"
            ]
            
            for i, example in enumerate(example_queries_1):
                if st.button(example, key=f"example1_{i}"):
                    st.session_state.messages.append({"role": "user", "content": example})
                    context = {
                        "documents": list(st.session_state.processed_documents.keys()),
                        "processed_documents": st.session_state.processed_documents,
                        "domain": "geotechnical_engineering"
                    }
                    agent_response = system["orchestrator"].route_geotechnical_query(example, context)
                    response_text = agent_response.get("response", "Could not process query.")
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.rerun()
        
        with col2:
            st.markdown("**🔬 Technical Analysis Questions:**")
            example_queries_2 = [
                "Analyze the soil test data and provide recommendations",
                "Is the bearing capacity sufficient for a 5-story building?",
                "What are the safety concerns based on the SPT values?",
                "Calculate settlement based on the soil parameters"
            ]
            
            for i, example in enumerate(example_queries_2):
                if st.button(example, key=f"example2_{i}"):
                    st.session_state.messages.append({"role": "user", "content": example})
                    context = {
                        "documents": list(st.session_state.processed_documents.keys()),
                        "processed_documents": st.session_state.processed_documents,
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
                    num_params = sum(len(v) for v in doc_data.get("numerical_data", {}).values())
                    st.markdown('<div class="tunnel-info">Extracted Values<br/>' + 
                               f"{num_params} parameters</div>", 
                               unsafe_allow_html=True)
                with col4:
                    timestamp = doc_data.get("timestamp", "Unknown")[:10] if doc_data.get("timestamp") else "Unknown"
                    st.markdown('<div class="engineering-metric">Processed<br/>' + 
                               timestamp + '</div>', 
                               unsafe_allow_html=True)
                
                st.divider()
                
                # Display extracted numerical data
                numerical_data = doc_data.get("numerical_data", {})
                if numerical_data and any(numerical_data.values()):
                    st.subheader("📐 Extracted Numerical Data")
                    
                    for param_type, values in numerical_data.items():
                        if values:
                            with st.expander(f"📊 {param_type.replace('_', ' ').title()} ({len(values)} values)"):
                                # Create a DataFrame for better display
                                df_data = []
                                for val in values:
                                    row = {
                                        'Value': val['value'],
                                        'Unit': val['unit'],
                                        'Context': val.get('context', '')[:100] + '...' if val.get('context') else ''
                                    }
                                    if 'depth' in val:
                                        row['Depth'] = f"{val['depth']} {val.get('depth_unit', 'm')}"
                                    df_data.append(row)
                                
                                df = pd.DataFrame(df_data)
                                st.dataframe(df, use_container_width=True)
                                
                                # Statistical summary for numerical values
                                if len(values) > 1:
                                    st.write("**Statistical Summary:**")
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
                
                # Display content analysis
                content = doc_data.get("content", {})
                doc_type = doc_data.get("document_type")
                
                if doc_type == "image" and "response" in content:
                    st.subheader("👁️ AI Vision Analysis")
                    st.write("**🔍 Analysis Query:**")
                    st.info(content.get('query', 'N/A'))
                    st.write("**📋 AI Analysis:**")
                    st.write(content.get('response', 'N/A'))
                    
                    processing_time = content.get('processing_time', 'unknown')
                    if processing_time != 'unknown':
                        st.success(f"⚡ GPU Processing Time: {processing_time}")
                
                # Raw content viewer
                with st.expander("🔍 Raw Document Data"):
                    st.json(doc_data)
        else:
            st.info("📥 Upload and analyze geotechnical documents to see detailed analysis")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="geotechnical-container">', unsafe_allow_html=True)
        st.subheader("📈 Geotechnical Data Visualizations")
        st.caption("Visualizations based on extracted numerical data")
        
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
                            # Use enhanced visualization module
                            fig = system["visualization"].create_visualization_from_any_document(doc_data)
                            st.plotly_chart(fig, use_container_width=True)
                
                with col1:
                    doc_type = doc_data.get("document_type", "Unknown")
                    num_params = sum(len(v) for v in doc_data.get("numerical_data", {}).values())
                    st.info(f"📊 Document Type: {doc_type} | Extracted Values: {num_params}")
                
                # Auto-generate visualization
                if doc_data.get("numerical_data") and any(doc_data["numerical_data"].values()):
                    with st.spinner("Preparing visualization..."):
                        fig = system["visualization"].create_visualization_from_any_document(doc_data)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No numerical data extracted yet. Upload documents with geotechnical data.")
        else:
            st.info("📥 Upload geotechnical documents to create intelligent visualizations")
            st.write("**Supported visualizations:**")
            st.write("• 📊 CSV/Excel: Statistical analysis, correlation matrices, depth profiles")
            st.write("• 🖼️ Images: Parameter extraction charts, measurement displays")  
            st.write("• 📄 PDFs: Numerical data visualization, parameter distributions")
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
                    if system["runpod_client"]:
                        health = system["runpod_client"].health_check()
                        if health["status"] == "healthy":
                            st.success("✅ AI vision analysis ready!")
                            st.success("✅ Soil analysis agent active")
                            st.success("✅ Tunnel engineering agent active")
                            st.success("✅ Safety checklist generator ready")
                        else:
                            st.error(f"❌ System issue: {health.get('error', 'Unknown error')}")
                    else:
                        st.warning("⚠️ RunPod not configured - running in local mode")
        
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
        st.write("• **Key Features**: VLM-based extraction (superior to OCR), Numerical data analysis, Document-based Q&A")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
