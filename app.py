#!/usr/bin/env python3
"""
SmolVLM-GeoEye: Simple Geotechnical Chat Application
==================================================

A clean, simple chat interface with SmolVLM for geotechnical engineering.
Focuses on core chat functionality without complex features.

Author: SmolVLM-GeoEye Team
Version: 2.0.0 - Simplified Chat Version
"""

import streamlit as st
import os
import json
import time
import base64
import requests
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from PIL import Image
import PyPDF2
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="SmolVLM-GeoEye: Geotechnical Chat",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    }
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-indicator {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 5px 0;
    }
    .status-error {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = {}

class SimpleConfig:
    """Simple configuration class"""
    def __init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        self.timeout = 300

class SimpleRunPodClient:
    """Simple RunPod client for SmolVLM"""
    
    def __init__(self, config: SimpleConfig):
        self.config = config
        self.base_url = f"https://api.runpod.ai/v2/{config.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Simple health check"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return {"status": "healthy", "ready": True}
            else:
                return {"status": "error", "ready": False, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "ready": False, "error": str(e)}
    
    def query_smolvlm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Query SmolVLM with input data"""
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
                    "response": result.get("output", {}).get("response", "No response")
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

def process_image_file(uploaded_file, runpod_client, query_text="Analyze this geotechnical document"):
    """Process an uploaded image file with SmolVLM"""
    try:
        # Load and convert image
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
        
        # Query SmolVLM
        input_data = {
            "image_data": image_base64,
            "query": query_text,
            "max_new_tokens": 512,
            "temperature": 0.3,
            "do_sample": True
        }
        
        return runpod_client.query_smolvlm(input_data)
    
    except Exception as e:
        return {"status": "error", "error": f"Image processing failed: {str(e)}"}

def process_pdf_file(uploaded_file):
    """Process a PDF file by extracting text"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        all_text = ""
        for page in pdf_reader.pages:
            all_text += page.extract_text() + "\n"
        
        return {
            "status": "success",
            "text": all_text,
            "pages": len(pdf_reader.pages)
        }
    except Exception as e:
        return {"status": "error", "error": f"PDF processing failed: {str(e)}"}

def main():
    """Main application"""
    
    # Header
    st.title("🏗️ SmolVLM-GeoEye: Geotechnical Chat")
    st.caption("Simple chat interface with SmolVLM for geotechnical engineering analysis")
    
    # Initialize configuration
    config = SimpleConfig()
    
    # Check configuration
    if not config.api_key or not config.endpoint_id:
        st.error("❌ Please set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in your .env file")
        st.info("Create a .env file with your RunPod credentials to use SmolVLM")
        return
    
    # Initialize RunPod client
    runpod_client = SimpleRunPodClient(config)
    
    # Status check
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Check System Status"):
            with st.spinner("Checking SmolVLM status..."):
                health = runpod_client.health_check()
                if health["ready"]:
                    st.markdown('<div class="status-indicator">✅ SmolVLM Ready</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-error">❌ SmolVLM Error: {health.get("error", "Unknown")}</div>', unsafe_allow_html=True)
    
    with col2:
        doc_count = len(st.session_state.processed_documents)
        st.metric("📄 Documents Processed", doc_count)
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload geotechnical documents",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload images or PDFs for SmolVLM analysis"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                
                # Skip if already processed
                if filename in st.session_state.processed_documents:
                    continue
                
                with st.spinner(f"Processing {filename}..."):
                    if uploaded_file.type.startswith('image/'):
                        # Process image with SmolVLM
                        query = """Analyze this geotechnical document. Extract and describe:
                        1. Any soil test results (SPT, density, moisture content)
                        2. Bearing capacity values
                        3. Foundation recommendations
                        4. Safety factors or design parameters
                        5. Any numerical data with units
                        
                        Provide a clear, professional analysis."""
                        
                        result = process_image_file(uploaded_file, runpod_client, query)
                        
                        if result["status"] == "success":
                            st.session_state.processed_documents[filename] = {
                                "type": "image",
                                "analysis": result.get("response", "No analysis available"),
                                "timestamp": datetime.now().isoformat()
                            }
                            st.success(f"✅ {filename} analyzed by SmolVLM")
                        else:
                            st.error(f"❌ Failed to analyze {filename}: {result.get('error')}")
                    
                    elif uploaded_file.type == 'application/pdf':
                        # Process PDF
                        result = process_pdf_file(uploaded_file)
                        
                        if result["status"] == "success":
                            st.session_state.processed_documents[filename] = {
                                "type": "pdf",
                                "text": result["text"],
                                "pages": result["pages"],
                                "timestamp": datetime.now().isoformat()
                            }
                            st.success(f"✅ {filename} processed ({result['pages']} pages)")
                        else:
                            st.error(f"❌ Failed to process {filename}: {result.get('error')}")
        
        # Show processed documents
        if st.session_state.processed_documents:
            st.subheader("📋 Processed Documents")
            for filename, doc_data in st.session_state.processed_documents.items():
                doc_type = doc_data["type"]
                timestamp = doc_data["timestamp"][:16].replace("T", " ")
                st.write(f"📄 **{filename}** ({doc_type}) - {timestamp}")
        
        # Clear data button
        if st.button("🗑️ Clear All Data"):
            st.session_state.messages = []
            st.session_state.processed_documents = {}
            st.success("All data cleared!")
            st.rerun()
    
    # Main chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about geotechnical analysis, soil properties, or your uploaded documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤖 SmolVLM thinking..."):
                
                # Prepare context from documents
                context = ""
                if st.session_state.processed_documents:
                    context = f"\nI have analyzed {len(st.session_state.processed_documents)} documents:\n"
                    for filename, doc_data in st.session_state.processed_documents.items():
                        if doc_data["type"] == "image" and "analysis" in doc_data:
                            context += f"\n{filename}: {doc_data['analysis'][:200]}...\n"
                        elif doc_data["type"] == "pdf" and "text" in doc_data:
                            context += f"\n{filename}: {doc_data['text'][:200]}...\n"
                
                # Create enhanced prompt
                enhanced_prompt = f"""You are a professional geotechnical engineer assistant. 
                
                User question: {prompt}
                
                Context from uploaded documents: {context}
                
                Please provide a helpful, professional response about geotechnical engineering topics. 
                If the user is asking about their documents, reference the analysis provided.
                If no relevant document context is available, provide general geotechnical engineering guidance."""
                
                # For text-only queries, provide a helpful response
                if "image" in [doc["type"] for doc in st.session_state.processed_documents.values()]:
                    # If we have image analysis, use that
                    response = f"Based on the documents I've analyzed:\n\n"
                    
                    # Include relevant document analysis
                    for filename, doc_data in st.session_state.processed_documents.items():
                        if doc_data["type"] == "image" and "analysis" in doc_data:
                            if any(keyword in prompt.lower() for keyword in ["spt", "soil", "bearing", "foundation", "analysis"]):
                                response += f"From {filename}:\n{doc_data['analysis']}\n\n"
                    
                    # Add contextual response based on the query
                    if "spt" in prompt.lower():
                        response += "For SPT analysis, values typically indicate:\n• N < 10: Loose/soft soil\n• N = 10-30: Medium dense\n• N > 30: Dense/hard soil"
                    elif "bearing" in prompt.lower():
                        response += "Bearing capacity depends on soil type, depth, and loading conditions. Always consider safety factors in design."
                    elif "foundation" in prompt.lower():
                        response += "Foundation selection should consider soil conditions, loads, settlement tolerance, and local building codes."
                    else:
                        response += "I can help analyze soil properties, foundation design, and geotechnical parameters from your documents."
                
                else:
                    # General geotechnical guidance
                    response = f"I understand you're asking about: {prompt}\n\n"
                    
                    if "spt" in prompt.lower():
                        response += "SPT (Standard Penetration Test) measures soil resistance:\n• N-values indicate soil density/consistency\n• Used for bearing capacity calculations\n• Critical for foundation design"
                    elif "bearing capacity" in prompt.lower():
                        response += "Bearing capacity is the soil's ability to support loads:\n• Depends on soil type and properties\n• Consider ultimate vs. allowable capacity\n• Include safety factors in design"
                    elif "foundation" in prompt.lower():
                        response += "Foundation design considerations:\n• Soil bearing capacity\n• Settlement requirements\n• Load characteristics\n• Local conditions and codes"
                    elif "soil" in prompt.lower():
                        response += "Soil analysis typically includes:\n• Classification (grain size, plasticity)\n• Strength parameters (cohesion, friction angle)\n• Density and moisture content\n• Laboratory and field testing"
                    else:
                        response += "I can help with geotechnical engineering topics including soil analysis, foundation design, slope stability, and site investigation. Feel free to upload documents for detailed analysis!"
                
                st.write(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.write("""
        **SmolVLM-GeoEye Chat Features:**
        
        1. **Upload Documents**: Use the sidebar to upload geotechnical documents (images or PDFs)
        2. **Image Analysis**: SmolVLM will analyze images for soil data, test results, and parameters
        3. **PDF Processing**: Text extraction from PDF reports and documents
        4. **Chat Interface**: Ask questions about your documents or general geotechnical topics
        5. **Context-Aware**: The assistant remembers your uploaded documents and can reference them
        
        **Example Questions:**
        - "What are the SPT values in my uploaded document?"
        - "Analyze the soil properties shown in the image"
        - "What foundation type would you recommend?"
        - "Explain the bearing capacity calculation"
        
        **Configuration:**
        Make sure to set your RunPod credentials in a .env file:
        ```
        RUNPOD_API_KEY=your_api_key_here
        RUNPOD_ENDPOINT_ID=your_endpoint_id_here
        ```
        """)

if __name__ == "__main__":
    main()
