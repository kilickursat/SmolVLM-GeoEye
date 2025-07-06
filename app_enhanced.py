#!/usr/bin/env python3
"""
SmolVLM-GeoEye: Production-Ready Geotechnical Engineering Application
====================================================================

A comprehensive geotechnical engineering workflow application powered by SmolVLM.
Features document analysis, AI agents, visualization, and cost tracking.

Author: SmolVLM-GeoEye Team
Version: 3.1.0 - Production Release
"""

import streamlit as st
import os
import json
import time
import base64
import tempfile
import hashlib
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

# Third-party imports
from PIL import Image
import PyPDF2
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Import all custom modules
from modules.config import get_config, ProductionConfig
from modules.smolvlm_client import EnhancedRunPodClient
from modules.data_extraction import EnhancedGeotechnicalDataExtractor
from modules.visualization import GeotechnicalVisualizationEngine
from modules.agents import GeotechnicalAgentOrchestrator
from modules.database import DatabaseManager
from modules.cache import CacheManager, DocumentCache, MetricsCache
from modules.monitoring import MetricsCollector, create_monitoring_middleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="SmolVLM-GeoEye: Geotechnical Engineering AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/kilickursat/SmolVLM-GeoEye',
        'Report a bug': 'https://github.com/kilickursat/SmolVLM-GeoEye/issues',
        'About': 'SmolVLM-GeoEye v3.1.0 - AI-Powered Geotechnical Engineering'
    }
)

# Enhanced CSS styling
st.markdown("""
<style>
    /* Main Theme */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Enhanced Status Cards */
    .status-card {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 5px;
    }
    
    /* SmolVLM Branding */
    .smolvlm-indicator {
        background: linear-gradient(135deg, #7F00FF 0%, #E100FF 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    /* Cost Tracker */
    .cost-tracker {
        background: linear-gradient(135deg, #fa8231 0%, #fd7014 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    
    /* Success Indicator */
    .success-indicator {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    
    /* Error Indicator */
    .error-indicator {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    
    /* Chat Container */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border: 2px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* Geotechnical Data Style */
    .geo-data-container {
        background: linear-gradient(135deg, #3a7bd5 0%, #3a6073 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.processed_documents = {}
    st.session_state.async_jobs = {}
    st.session_state.total_cost = 0.0
    st.session_state.smolvlm_queries = 0
    st.session_state.system_health = {"status": "unknown"}

class SmolVLMGeoEyeApp:
    """Main application class"""
    
    def __init__(self):
        """Initialize the application with all components"""
        # Load configuration
        self.config = get_config(ProductionConfig if os.getenv("ENVIRONMENT") == "production" else None)
        
        # Initialize components
        self.init_components()
        
        # Mark as initialized
        st.session_state.initialized = True
        
    def init_components(self):
        """Initialize all application components"""
        try:
            # Database
            self.db_manager = DatabaseManager(self.config.database_url)
            
            # Cache
            self.cache_manager = CacheManager(self.config)
            self.doc_cache = DocumentCache(self.cache_manager)
            self.metrics_cache = MetricsCache(self.cache_manager)
            
            # Monitoring
            self.metrics_collector = MetricsCollector(self.config)
            
            # RunPod Client
            self.runpod_client = EnhancedRunPodClient(self.config)
            
            # Data Extraction
            self.data_extractor = EnhancedGeotechnicalDataExtractor()
            
            # Visualization
            self.visualization_engine = GeotechnicalVisualizationEngine()
            
            # AI Agents
            self.agent_orchestrator = GeotechnicalAgentOrchestrator()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            st.error(f"Failed to initialize application: {str(e)}")
            raise
    
    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """Process an uploaded file with caching and monitoring"""
        filename = uploaded_file.name
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        
        # Check cache first
        cached_result = self.doc_cache.get_extracted_data(file_hash)
        if cached_result:
            self.metrics_collector.record_cache_access("document", hit=True)
            return cached_result
        
        self.metrics_collector.record_cache_access("document", hit=False)
        
        # Save to database
        doc_id = self.db_manager.save_document(
            filename=filename,
            document_type=uploaded_file.type.split('/')[0],
            file_size=len(uploaded_file.getvalue()),
            file_hash=file_hash
        )
        
        start_time = time.time()
        result = {"document_id": doc_id, "filename": filename, "file_hash": file_hash}
        
        try:
            if uploaded_file.type.startswith('image/'):
                result.update(self.process_image_file(uploaded_file, doc_id))
            elif uploaded_file.type == 'application/pdf':
                result.update(self.process_pdf_file(uploaded_file, doc_id))
            elif uploaded_file.name.endswith(('.csv', '.xlsx')):
                result.update(self.process_data_file(uploaded_file, doc_id))
            else:
                result.update(self.process_text_file(uploaded_file, doc_id))
            
            # Update processing status
            processing_time = time.time() - start_time
            self.db_manager.update_document_status(doc_id, "completed", processing_time)
            
            # Cache result
            self.doc_cache.set_extracted_data(file_hash, result)
            
            # Record metrics
            self.metrics_collector.record_document_processed(
                result.get("document_type", "unknown"), 
                success=True
            )
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            self.db_manager.update_document_status(doc_id, "failed", error_message=str(e))
            self.metrics_collector.record_document_processed(
                uploaded_file.type.split('/')[0], 
                success=False
            )
            result["error"] = str(e)
            result["processing_status"] = "failed"
        
        return result
    
    def process_image_file(self, uploaded_file, doc_id: int) -> Dict[str, Any]:
        """Process image file with SmolVLM"""
        # Convert image
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            image.save(tmp_file.name, 'PNG')
            with open(tmp_file.name, 'rb') as f:
                image_data = f.read()
            os.unlink(tmp_file.name)
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Query SmolVLM
        query = """Analyze this geotechnical engineering document. Extract and describe:
        1. All numerical data with units (SPT values, bearing capacity, density, etc.)
        2. Soil/rock properties and classifications
        3. Test results and measurements
        4. Engineering recommendations or design parameters
        5. Safety factors or critical values
        
        Provide a detailed, professional analysis with all numerical values clearly identified."""
        
        input_data = {
            "image_data": image_base64,
            "query": query,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "do_sample": self.config.do_sample
        }
        
        # Check cache
        cache_key = f"vision:{doc_id}:{hashlib.md5(query.encode()).hexdigest()}"
        cached_response = self.cache_manager.get(cache_key)
        
        if cached_response:
            response = cached_response
        else:
            # Run inference
            result = self.runpod_client.run_sync_with_tracking(input_data)
            
            if result["status"] == "success":
                response = result["response"]
                self.cache_manager.set(cache_key, response, ttl=7200)
                
                # Update costs
                st.session_state.total_cost += result.get("metrics", {}).get("cost_estimate", 0)
                st.session_state.smolvlm_queries += 1
                
                # Save AI analysis
                self.db_manager.save_ai_analysis(
                    document_id=doc_id,
                    query=query,
                    response=response,
                    model_name=self.config.model_name,
                    processing_time=result.get("metrics", {}).get("duration_seconds", 0),
                    cost_estimate=result.get("metrics", {}).get("cost_estimate", 0)
                )
            else:
                raise Exception(f"SmolVLM error: {result.get('error', 'Unknown error')}")
        
        # Extract numerical data
        extracted_data = self.data_extractor.extract_numerical_data_from_text(response)
        
        # Save extracted data
        if extracted_data:
            self.db_manager.save_numerical_data(doc_id, extracted_data)
        
        return {
            "document_type": "image",
            "processing_status": "completed",
            "content": {
                "query": query,
                "response": response,
                "processing_time": f"{time.time() - time.time():.2f}s"
            },
            "numerical_data": extracted_data,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_pdf_file(self, uploaded_file, doc_id: int) -> Dict[str, Any]:
        """Process PDF file"""
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        all_text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            all_text += f"\n--- Page {page_num + 1} ---\n"
            all_text += page.extract_text()
        
        # Extract numerical data
        extracted_data = self.data_extractor.extract_numerical_data_from_text(all_text)
        
        # Save extracted data
        if extracted_data:
            self.db_manager.save_numerical_data(doc_id, extracted_data)
        
        return {
            "document_type": "pdf",
            "processing_status": "completed",
            "content": {
                "text": all_text[:1000] + "..." if len(all_text) > 1000 else all_text,
                "pages": len(pdf_reader.pages),
                "total_length": len(all_text)
            },
            "numerical_data": extracted_data,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_data_file(self, uploaded_file, doc_id: int) -> Dict[str, Any]:
        """Process CSV/Excel files"""
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Extract numerical data
        extracted_data = self.data_extractor.extract_from_structured_data(df)
        
        # Save extracted data
        if extracted_data:
            self.db_manager.save_numerical_data(doc_id, extracted_data)
        
        return {
            "document_type": "data",
            "processing_status": "completed",
            "content": {
                "shape": df.shape,
                "columns": list(df.columns),
                "preview": df.head().to_dict()
            },
            "numerical_data": extracted_data,
            "dataframe": df,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_text_file(self, uploaded_file, doc_id: int) -> Dict[str, Any]:
        """Process text files"""
        content = uploaded_file.read().decode('utf-8')
        
        # Extract numerical data
        extracted_data = self.data_extractor.extract_numerical_data_from_text(content)
        
        # Save extracted data
        if extracted_data:
            self.db_manager.save_numerical_data(doc_id, extracted_data)
        
        return {
            "document_type": "text",
            "processing_status": "completed",
            "content": {
                "text": content[:1000] + "..." if len(content) > 1000 else content,
                "total_length": len(content)
            },
            "numerical_data": extracted_data,
            "timestamp": datetime.now().isoformat()
        }
    
    def render_status_header(self):
        """Render the status header with real-time indicators"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # RunPod Status
            health = self.runpod_client.health_check()
            if health["ready"]:
                st.markdown(
                    f'<div class="success-indicator">🚀 RunPod Active<br>{health["workers"]["ready"]} workers</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="error-indicator">❌ RunPod Offline</div>',
                    unsafe_allow_html=True
                )
        
        with col2:
            # SmolVLM Usage
            st.markdown(
                f'<div class="smolvlm-indicator">🤖 SmolVLM<br>{st.session_state.smolvlm_queries} queries</div>',
                unsafe_allow_html=True
            )
        
        with col3:
            # Cost Tracker
            st.markdown(
                f'<div class="cost-tracker">💰 Cost<br>${st.session_state.total_cost:.4f}</div>',
                unsafe_allow_html=True
            )
        
        with col4:
            # Documents Processed
            doc_count = len(st.session_state.processed_documents)
            st.markdown(
                f'<div class="metric-card">📄 Documents<br>{doc_count}</div>',
                unsafe_allow_html=True
            )
    
    def render_sidebar(self):
        """Render sidebar with file upload and controls"""
        with st.sidebar:
            st.header("📁 Document Upload")
            
            uploaded_files = st.file_uploader(
                "Upload geotechnical documents",
                type=['pdf', 'png', 'jpg', 'jpeg', 'csv', 'xlsx', 'txt', 'json'],
                accept_multiple_files=True,
                help="Upload images, PDFs, or data files for analysis"
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.processed_documents:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            try:
                                result = self.process_uploaded_file(uploaded_file)
                                st.session_state.processed_documents[uploaded_file.name] = result
                                st.success(f"✅ {uploaded_file.name} processed successfully")
                            except Exception as e:
                                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            
            st.divider()
            
            # Processed Documents
            if st.session_state.processed_documents:
                st.subheader("📋 Processed Documents")
                for filename, doc_data in st.session_state.processed_documents.items():
                    status = doc_data.get("processing_status", "unknown")
                    icon = "✅" if status == "completed" else "❌"
                    
                    with st.expander(f"{icon} {filename}"):
                        st.write(f"Type: {doc_data.get('document_type', 'unknown')}")
                        st.write(f"Status: {status}")
                        
                        # Show extracted data summary
                        numerical_data = doc_data.get("numerical_data", {})
                        if numerical_data:
                            st.write("**Extracted Parameters:**")
                            for param, values in numerical_data.items():
                                st.write(f"- {param}: {len(values)} values")
            
            st.divider()
            
            # Controls
            st.subheader("🎛️ Controls")
            
            if st.button("🔄 Refresh System Status"):
                health = self.runpod_client.enhanced_health_check()
                st.session_state.system_health = health
                st.success("System status updated")
            
            if st.button("📊 Generate Report"):
                self.generate_comprehensive_report()
            
            if st.button("🗑️ Clear All Data"):
                st.session_state.messages = []
                st.session_state.processed_documents = {}
                st.session_state.total_cost = 0.0
                st.session_state.smolvlm_queries = 0
                self.cache_manager.clear()
                st.success("All data cleared!")
                st.rerun()
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about geotechnical analysis, soil properties, or your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤖 SmolVLM thinking..."):
                    # Prepare context
                    context = {
                        "processed_documents": st.session_state.processed_documents,
                        "domain": "geotechnical_engineering"
                    }
                    
                    # Get agent response
                    agent_response = self.agent_orchestrator.route_query(prompt, context)
                    
                    # Display response
                    st.write(agent_response.response)
                    
                    # Show recommendations and warnings
                    if agent_response.recommendations:
                        st.info("**Recommendations:**\n" + "\n".join(f"- {r}" for r in agent_response.recommendations))
                    
                    if agent_response.warnings:
                        st.warning("**Warnings:**\n" + "\n".join(f"- {w}" for w in agent_response.warnings))
                    
                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": agent_response.response
                    })
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_data_analysis_tab(self):
        """Render data analysis tab"""
        st.subheader("📊 Geotechnical Data Analysis")
        
        if not st.session_state.processed_documents:
            st.info("📥 Upload documents to see detailed analysis")
            return
        
        # Document selector
        doc_names = list(st.session_state.processed_documents.keys())
        selected_doc = st.selectbox("Select document:", doc_names)
        
        if selected_doc:
            doc_data = st.session_state.processed_documents[selected_doc]
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Document Type", doc_data.get("document_type", "Unknown"))
            
            with col2:
                st.metric("Processing Status", doc_data.get("processing_status", "Unknown"))
            
            with col3:
                numerical_data = doc_data.get("numerical_data", {})
                total_values = sum(len(v) for v in numerical_data.values())
                st.metric("Extracted Values", total_values)
            
            with col4:
                if "timestamp" in doc_data:
                    st.metric("Processed", doc_data["timestamp"][:10])
            
            # Display extracted data
            if numerical_data:
                st.subheader("📐 Extracted Numerical Data")
                
                # Get statistical summary
                summary = self.data_extractor.get_statistical_summary(numerical_data)
                
                # Display summary for each parameter
                for param_type, stats in summary.items():
                    with st.expander(f"📊 {param_type.replace('_', ' ').title()}"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Count", stats['count'])
                        with col2:
                            st.metric("Min", f"{stats['min']:.2f}")
                        with col3:
                            st.metric("Max", f"{stats['max']:.2f}")
                        with col4:
                            st.metric("Mean", f"{stats['mean']:.2f}")
                        
                        # Show individual values
                        if st.checkbox(f"Show all {param_type} values", key=f"show_{param_type}"):
                            df = pd.DataFrame([asdict(v) for v in numerical_data[param_type]])
                            st.dataframe(df, use_container_width=True)
            
            # AI Analysis
            if doc_data.get("document_type") == "image" and "content" in doc_data:
                st.subheader("🤖 SmolVLM Analysis")
                st.write(doc_data["content"].get("response", "No analysis available"))
    
    def render_visualization_tab(self):
        """Render visualization tab"""
        st.subheader("📈 Geotechnical Data Visualizations")
        
        if not st.session_state.processed_documents:
            st.info("📥 Upload documents to create visualizations")
            return
        
        # Document selector
        doc_names = list(st.session_state.processed_documents.keys())
        selected_doc = st.selectbox("Select document for visualization:", doc_names, key="viz_doc")
        
        if selected_doc:
            doc_data = st.session_state.processed_documents[selected_doc]
            
            # Generate visualizations
            if doc_data.get("numerical_data"):
                # Main visualization
                fig = self.visualization_engine.create_visualization_from_any_document(doc_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Parameter Distribution"):
                        dist_fig = self.visualization_engine.create_parameter_distribution(
                            doc_data["numerical_data"]
                        )
                        st.plotly_chart(dist_fig, use_container_width=True)
                
                with col2:
                    if st.button("🔗 Correlation Matrix"):
                        corr_fig = self.visualization_engine.create_correlation_matrix(
                            doc_data["numerical_data"]
                        )
                        st.plotly_chart(corr_fig, use_container_width=True)
                
                # Export options
                if st.checkbox("Export visualization"):
                    format = st.selectbox("Format:", ["png", "html", "svg"])
                    if st.button("Download"):
                        filename = f"{selected_doc}_visualization.{format}"
                        self.visualization_engine.export_figure(fig, filename, format)
                        st.success(f"Exported to {filename}")
            else:
                st.warning("No numerical data available for visualization")
    
    def render_system_tab(self):
        """Render system status tab"""
        st.subheader("🚀 System Status & Performance")
        
        # Get current health
        health = self.runpod_client.enhanced_health_check()
        
        # System Status
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔧 System Configuration:**")
            st.write(f"- RunPod Configured: {'✅' if self.config.api_key else '❌'}")
            st.write(f"- Environment: {os.getenv('ENVIRONMENT', 'development')}")
            st.write(f"- Cache Enabled: {'✅' if self.config.cache_enabled else '❌'}")
            st.write(f"- Cost Tracking: {'✅' if self.config.cost_tracking_enabled else '❌'}")
        
        with col2:
            st.write("**📊 Performance Metrics:**")
            if health["status"] == "healthy":
                st.write(f"- Workers Ready: {health['workers']['ready']}")
                st.write(f"- Success Rate: {health['metrics']['success_rate']:.1f}%")
                st.write(f"- Avg Response: {health['metrics']['avg_response_time_ms']:.0f}ms")
                st.write(f"- Cost/Hour: ${health['cost_per_hour']:.4f}")
        
        # Detailed Metrics
        st.subheader("📈 Detailed Metrics")
        
        # Usage statistics
        usage_stats = self.runpod_client.get_usage_statistics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Jobs", usage_stats["total_jobs"])
            st.metric("Success Rate", f"{usage_stats['success_rate']:.1f}%")
        
        with col2:
            st.metric("Total Cost", f"${usage_stats['total_cost']:.4f}")
            st.metric("Cost per Job", f"${usage_stats['cost_per_job']:.4f}")
        
        with col3:
            st.metric("Avg Response Time", f"{usage_stats['avg_response_time']*1000:.0f}ms")
            st.metric("Hourly Cost", f"${usage_stats['hourly_cost']:.4f}")
        
        # Recommendations
        if health.get("recommendations"):
            st.subheader("🎯 Optimization Recommendations")
            for rec in health["recommendations"]:
                st.info(rec)
        
        # System Health
        st.subheader("💚 System Health")
        system_health = self.metrics_collector.create_health_check()
        
        for check_name, check_data in system_health["checks"].items():
            status_icon = "✅" if check_data["status"] == "pass" else "⚠️" if check_data["status"] == "warn" else "❌"
            st.write(f"{status_icon} **{check_name.replace('_', ' ').title()}**: {check_data['status']}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        if not st.session_state.processed_documents:
            st.warning("No documents to analyze")
            return
        
        with st.spinner("Generating comprehensive report..."):
            # Get all analyses
            all_analyses = self.agent_orchestrator.get_comprehensive_analysis({
                "processed_documents": st.session_state.processed_documents
            })
            
            # Create report
            report = "# SmolVLM-GeoEye Comprehensive Report\n\n"
            report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # Executive Summary
            report += "## Executive Summary\n\n"
            report += f"- Documents Analyzed: {len(st.session_state.processed_documents)}\n"
            report += f"- SmolVLM Queries: {st.session_state.smolvlm_queries}\n"
            report += f"- Total Cost: ${st.session_state.total_cost:.4f}\n\n"
            
            # Document Summary
            report += "## Document Analysis Summary\n\n"
            for filename, doc_data in st.session_state.processed_documents.items():
                report += f"### {filename}\n"
                report += f"- Type: {doc_data.get('document_type', 'unknown')}\n"
                report += f"- Status: {doc_data.get('processing_status', 'unknown')}\n"
                
                numerical_data = doc_data.get('numerical_data', {})
                if numerical_data:
                    report += "- Extracted Parameters:\n"
                    for param, values in numerical_data.items():
                        report += f"  - {param}: {len(values)} values\n"
                report += "\n"
            
            # Agent Analyses
            report += "## Expert Analysis\n\n"
            for agent_type, analysis in all_analyses.items():
                report += f"### {analysis.agent_type}\n\n"
                report += analysis.response + "\n\n"
                
                if analysis.recommendations:
                    report += "**Recommendations:**\n"
                    for rec in analysis.recommendations:
                        report += f"- {rec}\n"
                    report += "\n"
                
                if analysis.warnings:
                    report += "**Warnings:**\n"
                    for warn in analysis.warnings:
                        report += f"- {warn}\n"
                    report += "\n"
            
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"SmolVLM_GeoEye_Report_{timestamp}.md"
            
            with open(report_filename, 'w') as f:
                f.write(report)
            
            st.success(f"Report generated: {report_filename}")
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=report_filename,
                mime="text/markdown"
            )
    
    def run(self):
        """Run the main application"""
        # Header
        st.title("🏗️ SmolVLM-GeoEye: Geotechnical Engineering AI")
        st.caption("Production-ready AI workflow for geotechnical analysis powered by SmolVLM")
        
        # Status header
        self.render_status_header()
        
        # Sidebar
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 AI Chat",
            "📊 Data Analysis",
            "📈 Visualizations",
            "🚀 System Status"
        ])
        
        with tab1:
            self.render_chat_interface()
        
        with tab2:
            self.render_data_analysis_tab()
        
        with tab3:
            self.render_visualization_tab()
        
        with tab4:
            self.render_system_tab()

def main():
    """Main entry point"""
    try:
        # Create and run application
        app = SmolVLMGeoEyeApp()
        app.run()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"Application error: {str(e)}")
        st.error("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
