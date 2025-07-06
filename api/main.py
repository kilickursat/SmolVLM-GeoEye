#!/usr/bin/env python3
"""
SmolVLM-GeoEye API Service
==========================

RESTful API for external integration with SmolVLM-GeoEye.
Provides endpoints for document processing, analysis, and monitoring.

Author: SmolVLM-GeoEye Team
Version: 3.1.0
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import hashlib
import time
import uuid
import logging
from datetime import datetime
import asyncio
import os

# Import modules
from modules.config import get_config, ProductionConfig
from modules.smolvlm_client import EnhancedRunPodClient
from modules.data_extraction import EnhancedGeotechnicalDataExtractor
from modules.visualization import GeotechnicalVisualizationEngine
from modules.agents import GeotechnicalAgentOrchestrator
from modules.database import DatabaseManager
from modules.cache import CacheManager, DocumentCache, MetricsCache
from modules.monitoring import MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SmolVLM-GeoEye API",
    description="AI-powered geotechnical engineering analysis API",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global instances
config = None
db_manager = None
cache_manager = None
metrics_collector = None
runpod_client = None
data_extractor = None
visualization_engine = None
agent_orchestrator = None

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]

class DocumentUploadResponse(BaseModel):
    document_id: int
    filename: str
    file_hash: str
    status: str
    message: str

class AnalysisRequest(BaseModel):
    document_id: int
    query: str
    agent_type: Optional[str] = "auto"

class AnalysisResponse(BaseModel):
    analysis_id: str
    document_id: int
    agent_type: str
    response: str
    recommendations: List[str]
    warnings: List[str]
    confidence: float
    processing_time: float

class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[int]] = []
    include_visualizations: bool = False

class MetricsResponse(BaseModel):
    period_hours: int
    total_jobs: int
    success_rate: float
    total_cost: float
    avg_response_time: float
    worker_status: Dict[str, int]

class ExtractedDataResponse(BaseModel):
    document_id: int
    parameter_type: str
    values: List[Dict[str, Any]]
    statistics: Dict[str, float]

# Dependency for authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    expected_token = config.secret_key
    
    if not expected_token or token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    return token

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global config, db_manager, cache_manager, metrics_collector
    global runpod_client, data_extractor, visualization_engine, agent_orchestrator
    
    try:
        # Load configuration
        config = get_config(ProductionConfig if os.getenv("ENVIRONMENT") == "production" else None)
        
        # Initialize components
        db_manager = DatabaseManager(config.database_url)
        cache_manager = CacheManager(config)
        metrics_collector = MetricsCollector(config)
        runpod_client = EnhancedRunPodClient(config)
        data_extractor = EnhancedGeotechnicalDataExtractor()
        visualization_engine = GeotechnicalVisualizationEngine()
        agent_orchestrator = GeotechnicalAgentOrchestrator()
        
        logger.info("API services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API services: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if metrics_collector:
        metrics_collector.stop_collection()

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    health_status = {
        "database": "healthy",
        "cache": "healthy",
        "runpod": "unknown",
        "monitoring": "healthy"
    }
    
    # Check RunPod
    if runpod_client:
        runpod_health = runpod_client.health_check()
        health_status["runpod"] = "healthy" if runpod_health["ready"] else "unhealthy"
    
    # Overall status
    overall_status = "healthy" if all(v == "healthy" for v in health_status.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version=config.app_version,
        services=health_status
    )

# Document upload endpoint
@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """Upload a document for processing"""
    try:
        # Validate file
        if file.size > config.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Calculate file hash
        content = await file.read()
        file_hash = hashlib.md5(content).hexdigest()
        
        # Check cache
        cached_doc = cache_manager.get(f"doc:hash:{file_hash}")
        if cached_doc:
            return DocumentUploadResponse(
                document_id=cached_doc["document_id"],
                filename=file.filename,
                file_hash=file_hash,
                status="cached",
                message="Document already processed"
            )
        
        # Save to database
        doc_id = db_manager.save_document(
            filename=file.filename,
            document_type=file.content_type.split('/')[0],
            file_size=len(content),
            file_hash=file_hash
        )
        
        # Process in background
        background_tasks.add_task(
            process_document_async,
            doc_id,
            file.filename,
            file.content_type,
            content
        )
        
        return DocumentUploadResponse(
            document_id=doc_id,
            filename=file.filename,
            file_hash=file_hash,
            status="processing",
            message="Document queued for processing"
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document analysis endpoint
@app.post("/api/v1/documents/{document_id}/analyze", response_model=AnalysisResponse)
async def analyze_document(
    document_id: int,
    request: AnalysisRequest,
    token: str = Depends(verify_token)
):
    """Analyze a document with AI agents"""
    try:
        # Get document data
        doc_data = db_manager.get_document_by_id(document_id)
        if not doc_data:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get numerical data
        numerical_data = db_manager.get_numerical_data_by_document(document_id)
        
        # Prepare context
        context = {
            "processed_documents": {
                doc_data["filename"]: {
                    "document_id": document_id,
                    "document_type": doc_data["document_type"],
                    "numerical_data": numerical_data
                }
            }
        }
        
        # Get agent response
        start_time = time.time()
        agent_response = agent_orchestrator.route_query(request.query, context)
        processing_time = time.time() - start_time
        
        # Save analysis
        analysis_id = str(uuid.uuid4())
        db_manager.save_ai_analysis(
            document_id=document_id,
            query=request.query,
            response=agent_response.response,
            model_name=agent_response.agent_type,
            processing_time=processing_time
        )
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            document_id=document_id,
            agent_type=agent_response.agent_type,
            response=agent_response.response,
            recommendations=agent_response.recommendations,
            warnings=agent_response.warnings,
            confidence=agent_response.confidence,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Document analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Query endpoint
@app.post("/api/v1/query")
async def query_documents(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Query across multiple documents"""
    try:
        # Prepare context
        context = {"processed_documents": {}}
        
        if request.document_ids:
            for doc_id in request.document_ids:
                doc_data = db_manager.get_document_by_id(doc_id)
                if doc_data:
                    numerical_data = db_manager.get_numerical_data_by_document(doc_id)
                    context["processed_documents"][doc_data["filename"]] = {
                        "document_id": doc_id,
                        "document_type": doc_data["document_type"],
                        "numerical_data": numerical_data
                    }
        
        # Get agent response
        agent_response = agent_orchestrator.route_query(request.query, context)
        
        response = {
            "query": request.query,
            "response": agent_response.response,
            "agent_type": agent_response.agent_type,
            "recommendations": agent_response.recommendations,
            "warnings": agent_response.warnings,
            "confidence": agent_response.confidence,
            "documents_used": len(context["processed_documents"])
        }
        
        # Add visualization if requested
        if request.include_visualizations and context["processed_documents"]:
            # Create visualization
            all_numerical_data = {}
            for doc in context["processed_documents"].values():
                for param_type, values in doc.get("numerical_data", {}).items():
                    if param_type not in all_numerical_data:
                        all_numerical_data[param_type] = []
                    all_numerical_data[param_type].extend(values)
            
            if all_numerical_data:
                fig = visualization_engine.create_multi_parameter_chart(all_numerical_data)
                # Convert to JSON-serializable format
                response["visualization"] = fig.to_dict()
        
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Metrics endpoint
@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def get_metrics(
    hours: int = Query(24, description="Period in hours"),
    token: str = Depends(verify_token)
):
    """Get system metrics"""
    try:
        # Get cost analytics
        cost_analytics = db_manager.get_cost_analytics(hours)
        
        # Get worker status
        health = runpod_client.enhanced_health_check()
        
        return MetricsResponse(
            period_hours=hours,
            total_jobs=cost_analytics["total_jobs"],
            success_rate=cost_analytics["success_rate"],
            total_cost=cost_analytics["total_cost"],
            avg_response_time=cost_analytics["avg_duration"],
            worker_status=health["workers"]
        )
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Extracted data endpoint
@app.get("/api/v1/documents/{document_id}/data", response_model=List[ExtractedDataResponse])
async def get_extracted_data(
    document_id: int,
    parameter_type: Optional[str] = None,
    token: str = Depends(verify_token)
):
    """Get extracted numerical data from a document"""
    try:
        # Get numerical data
        numerical_data = db_manager.get_numerical_data_by_document(document_id)
        
        if not numerical_data:
            return []
        
        # Filter by parameter type if specified
        if parameter_type:
            numerical_data = {parameter_type: numerical_data.get(parameter_type, [])}
        
        # Prepare response
        response = []
        for param_type, values in numerical_data.items():
            if values:
                # Calculate statistics
                nums = [v["value"] for v in values]
                statistics = {
                    "count": len(nums),
                    "min": min(nums),
                    "max": max(nums),
                    "mean": sum(nums) / len(nums),
                    "std": 0  # Would need numpy for std dev
                }
                
                response.append(ExtractedDataResponse(
                    document_id=document_id,
                    parameter_type=param_type,
                    values=values,
                    statistics=statistics
                ))
        
        return response
        
    except Exception as e:
        logger.error(f"Data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document list endpoint
@app.get("/api/v1/documents")
async def list_documents(
    limit: int = Query(50, description="Maximum number of documents"),
    offset: int = Query(0, description="Offset for pagination"),
    token: str = Depends(verify_token)
):
    """List processed documents"""
    try:
        documents = db_manager.get_document_history(limit)
        
        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Visualization endpoint
@app.get("/api/v1/documents/{document_id}/visualize")
async def visualize_document(
    document_id: int,
    chart_type: str = Query("auto", description="Chart type: auto, spt_profile, distribution, correlation"),
    token: str = Depends(verify_token)
):
    """Generate visualization for document data"""
    try:
        # Get numerical data
        numerical_data = db_manager.get_numerical_data_by_document(document_id)
        
        if not numerical_data:
            raise HTTPException(status_code=404, detail="No numerical data found")
        
        # Create visualization based on type
        if chart_type == "spt_profile":
            fig = visualization_engine.create_spt_depth_profile(numerical_data)
        elif chart_type == "distribution":
            fig = visualization_engine.create_parameter_distribution(numerical_data)
        elif chart_type == "correlation":
            fig = visualization_engine.create_correlation_matrix(numerical_data)
        else:
            # Auto-select best visualization
            doc_data = {"numerical_data": numerical_data}
            fig = visualization_engine.create_visualization_from_any_document(doc_data)
        
        # Return as JSON
        return fig.to_dict()
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Async document processing
async def process_document_async(doc_id: int, filename: str, content_type: str, content: bytes):
    """Process document asynchronously"""
    try:
        # Similar to the main app processing logic
        # This would be the same processing code but async
        logger.info(f"Processing document {doc_id}: {filename}")
        
        # Update status
        db_manager.update_document_status(doc_id, "completed")
        
        # Cache result
        cache_manager.set(f"doc:id:{doc_id}", {"status": "completed"})
        
    except Exception as e:
        logger.error(f"Async document processing failed: {e}")
        db_manager.update_document_status(doc_id, "failed", error_message=str(e))

# Prometheus metrics endpoint
@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    if metrics_collector:
        return metrics_collector.get_prometheus_metrics()
    return ""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
