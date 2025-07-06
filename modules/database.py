#!/usr/bin/env python3
"""
SmolVLM-GeoEye Database Module
==============================

Database management for persistent storage of geotechnical data,
analysis results, and system metrics.

Author: SmolVLM-GeoEye Team
Version: 3.1.0
"""

import logging
import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from contextlib import contextmanager
from threading import Lock
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

Base = declarative_base()

# Database Models
class Document(Base):
    """Document model for storing uploaded files metadata"""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), unique=True, nullable=False)
    document_type = Column(String(50))
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer)
    file_hash = Column(String(64))
    processing_status = Column(String(50), default='pending')
    processing_time = Column(Float)
    error_message = Column(Text)
    
    # Relationships
    numerical_data = relationship("NumericalData", back_populates="document", cascade="all, delete-orphan")
    ai_analyses = relationship("AIAnalysis", back_populates="document", cascade="all, delete-orphan")
    
class NumericalData(Base):
    """Numerical data extracted from documents"""
    __tablename__ = 'numerical_data'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    parameter_type = Column(String(100))
    value = Column(Float)
    unit = Column(String(50))
    depth = Column(Float, nullable=True)
    depth_unit = Column(String(20), nullable=True)
    context = Column(Text)
    confidence = Column(Float)
    extraction_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="numerical_data")

class AIAnalysis(Base):
    """AI analysis results from SmolVLM"""
    __tablename__ = 'ai_analyses'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    query = Column(Text)
    response = Column(Text)
    model_name = Column(String(100))
    processing_time = Column(Float)
    tokens_used = Column(Integer)
    cost_estimate = Column(Float)
    analysis_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="ai_analyses")

class JobMetrics(Base):
    """Job execution metrics"""
    __tablename__ = 'job_metrics'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(100), unique=True)
    job_type = Column(String(50))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)
    success = Column(Boolean)
    error_message = Column(Text)
    worker_id = Column(String(100))
    cost_estimate = Column(Float)
    
class SystemMetrics(Base):
    """System performance metrics"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_type = Column(String(50))
    metric_value = Column(Float)
    metric_unit = Column(String(20))
    additional_info = Column(Text)

class UserSession(Base):
    """User session tracking"""
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    documents_processed = Column(Integer, default=0)
    queries_made = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    user_agent = Column(String(255))

class DatabaseManager:
    """Manages database operations"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._engine = None
        self._session_factory = None
        self._lock = Lock()
        self.initialize()
    
    def initialize(self):
        """Initialize database connection and create tables"""
        try:
            # Create engine with connection pooling disabled for SQLite
            if 'sqlite' in self.database_url:
                self._engine = create_engine(
                    self.database_url,
                    connect_args={'check_same_thread': False},
                    poolclass=NullPool
                )
            else:
                self._engine = create_engine(self.database_url)
            
            # Create tables
            Base.metadata.create_all(self._engine)
            
            # Create session factory
            self._session_factory = sessionmaker(bind=self._engine)
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session context manager"""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def save_document(self, filename: str, document_type: str, 
                     file_size: int, file_hash: str) -> int:
        """Save document metadata"""
        with self.get_session() as session:
            # Check if document already exists
            existing = session.query(Document).filter_by(filename=filename).first()
            if existing:
                return existing.id
            
            document = Document(
                filename=filename,
                document_type=document_type,
                file_size=file_size,
                file_hash=file_hash,
                processing_status='uploaded'
            )
            session.add(document)
            session.flush()
            return document.id
    
    def update_document_status(self, document_id: int, status: str, 
                             processing_time: Optional[float] = None,
                             error_message: Optional[str] = None):
        """Update document processing status"""
        with self.get_session() as session:
            document = session.query(Document).get(document_id)
            if document:
                document.processing_status = status
                if processing_time:
                    document.processing_time = processing_time
                if error_message:
                    document.error_message = error_message
    
    def save_numerical_data(self, document_id: int, extracted_data: Dict[str, List[Any]]):
        """Save extracted numerical data"""
        with self.get_session() as session:
            # Clear existing data for re-processing
            session.query(NumericalData).filter_by(document_id=document_id).delete()
            
            # Save new data
            for param_type, values in extracted_data.items():
                for value_obj in values:
                    if hasattr(value_obj, '__dict__'):
                        # ExtractedValue object
                        data = NumericalData(
                            document_id=document_id,
                            parameter_type=param_type,
                            value=value_obj.value,
                            unit=value_obj.unit,
                            depth=value_obj.depth,
                            depth_unit=value_obj.depth_unit,
                            context=value_obj.context,
                            confidence=value_obj.confidence
                        )
                    else:
                        # Dictionary
                        data = NumericalData(
                            document_id=document_id,
                            parameter_type=param_type,
                            value=value_obj.get('value'),
                            unit=value_obj.get('unit', ''),
                            depth=value_obj.get('depth'),
                            depth_unit=value_obj.get('depth_unit'),
                            context=value_obj.get('context', ''),
                            confidence=value_obj.get('confidence', 0.5)
                        )
                    session.add(data)
    
    def save_ai_analysis(self, document_id: int, query: str, response: str,
                        model_name: str, processing_time: float,
                        tokens_used: int = 0, cost_estimate: float = 0.0):
        """Save AI analysis results"""
        with self.get_session() as session:
            analysis = AIAnalysis(
                document_id=document_id,
                query=query,
                response=response,
                model_name=model_name,
                processing_time=processing_time,
                tokens_used=tokens_used,
                cost_estimate=cost_estimate
            )
            session.add(analysis)
    
    def save_job_metrics(self, metrics: Dict[str, Any]):
        """Save job execution metrics"""
        with self.get_session() as session:
            job_metric = JobMetrics(
                job_id=metrics['job_id'],
                job_type=metrics.get('job_type', 'inference'),
                start_time=datetime.fromtimestamp(metrics['start_time']),
                end_time=datetime.fromtimestamp(metrics['end_time']),
                duration_seconds=metrics['duration_seconds'],
                success=metrics['success'],
                error_message=metrics.get('error'),
                worker_id=metrics.get('worker_id', 'unknown'),
                cost_estimate=metrics.get('cost_estimate', 0.0)
            )
            session.add(job_metric)
    
    def save_system_metrics(self, metric_type: str, value: float, 
                           unit: str = '', info: Optional[Dict] = None):
        """Save system performance metrics"""
        with self.get_session() as session:
            metric = SystemMetrics(
                metric_type=metric_type,
                metric_value=value,
                metric_unit=unit,
                additional_info=json.dumps(info) if info else None
            )
            session.add(metric)
    
    def get_document_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent document processing history"""
        with self.get_session() as session:
            documents = session.query(Document)\
                .order_by(Document.upload_timestamp.desc())\
                .limit(limit)\
                .all()
            
            return [{
                'id': doc.id,
                'filename': doc.filename,
                'document_type': doc.document_type,
                'upload_timestamp': doc.upload_timestamp.isoformat(),
                'processing_status': doc.processing_status,
                'processing_time': doc.processing_time,
                'numerical_data_count': len(doc.numerical_data),
                'ai_analyses_count': len(doc.ai_analyses)
            } for doc in documents]
    
    def get_numerical_data_by_document(self, document_id: int) -> Dict[str, List[Dict]]:
        """Get all numerical data for a document"""
        with self.get_session() as session:
            data = session.query(NumericalData)\
                .filter_by(document_id=document_id)\
                .all()
            
            result = {}
            for item in data:
                if item.parameter_type not in result:
                    result[item.parameter_type] = []
                
                result[item.parameter_type].append({
                    'value': item.value,
                    'unit': item.unit,
                    'depth': item.depth,
                    'depth_unit': item.depth_unit,
                    'context': item.context,
                    'confidence': item.confidence
                })
            
            return result
    
    def get_cost_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get cost analytics for specified time period"""
        with self.get_session() as session:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Get job metrics
            jobs = session.query(JobMetrics)\
                .filter(JobMetrics.start_time >= cutoff_time)\
                .all()
            
            if not jobs:
                return {
                    'period_hours': hours,
                    'total_jobs': 0,
                    'total_cost': 0.0,
                    'success_rate': 0.0,
                    'avg_duration': 0.0
                }
            
            total_jobs = len(jobs)
            successful_jobs = [j for j in jobs if j.success]
            total_cost = sum(j.cost_estimate or 0 for j in jobs)
            
            return {
                'period_hours': hours,
                'total_jobs': total_jobs,
                'successful_jobs': len(successful_jobs),
                'failed_jobs': total_jobs - len(successful_jobs),
                'total_cost': total_cost,
                'success_rate': len(successful_jobs) / total_jobs * 100,
                'avg_duration': sum(j.duration_seconds for j in jobs) / total_jobs,
                'avg_cost_per_job': total_cost / total_jobs,
                'hourly_rate': total_cost / hours
            }
    
    def get_parameter_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all parameter types"""
        with self.get_session() as session:
            # Raw SQL for aggregation
            query = """
                SELECT 
                    parameter_type,
                    COUNT(*) as count,
                    MIN(value) as min_value,
                    MAX(value) as max_value,
                    AVG(value) as avg_value,
                    AVG(confidence) as avg_confidence
                FROM numerical_data
                GROUP BY parameter_type
            """
            
            result = session.execute(query)
            
            stats = {}
            for row in result:
                stats[row[0]] = {
                    'count': row[1],
                    'min': row[2],
                    'max': row[3],
                    'mean': row[4],
                    'avg_confidence': row[5]
                }
            
            return stats
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data"""
        with self.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Delete old job metrics
            session.query(JobMetrics)\
                .filter(JobMetrics.start_time < cutoff_date)\
                .delete()
            
            # Delete old system metrics
            session.query(SystemMetrics)\
                .filter(SystemMetrics.timestamp < cutoff_date)\
                .delete()
            
            logger.info(f"Cleaned up data older than {days} days")
    
    def export_to_csv(self, output_dir: str):
        """Export database tables to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        with self.get_session() as session:
            # Export documents
            docs_query = "SELECT * FROM documents"
            docs_df = pd.read_sql(docs_query, session.bind)
            docs_df.to_csv(output_path / "documents.csv", index=False)
            
            # Export numerical data
            data_query = "SELECT * FROM numerical_data"
            data_df = pd.read_sql(data_query, session.bind)
            data_df.to_csv(output_path / "numerical_data.csv", index=False)
            
            # Export AI analyses
            ai_query = "SELECT * FROM ai_analyses"
            ai_df = pd.read_sql(ai_query, session.bind)
            ai_df.to_csv(output_path / "ai_analyses.csv", index=False)
            
            logger.info(f"Database exported to {output_path}")
