#!/usr/bin/env python3
"""
SmolVLM-GeoEye Monitoring Module
================================

Performance monitoring, metrics collection, and observability.
Integrates with Prometheus for production monitoring.

Author: SmolVLM-GeoEye Team
Version: 3.1.0
"""

import logging
import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('smolvlm_requests_total', 'Total number of requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('smolvlm_request_duration_seconds', 'Request duration', ['endpoint'])
ACTIVE_REQUESTS = Gauge('smolvlm_active_requests', 'Number of active requests')
WORKER_COUNT = Gauge('smolvlm_workers', 'Number of workers', ['status'])
INFERENCE_TIME = Histogram('smolvlm_inference_seconds', 'Inference duration')
DOCUMENT_PROCESSED = Counter('smolvlm_documents_processed', 'Documents processed', ['type', 'status'])
CACHE_HITS = Counter('smolvlm_cache_hits', 'Cache hit count', ['cache_type'])
CACHE_MISSES = Counter('smolvlm_cache_misses', 'Cache miss count', ['cache_type'])
ERROR_COUNT = Counter('smolvlm_errors', 'Error count', ['error_type'])
COST_GAUGE = Gauge('smolvlm_cost_usd', 'Current cost in USD')
SYSTEM_INFO = Info('smolvlm_system', 'System information')

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    timestamp: float
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str]

class MetricsCollector:
    """Collects and aggregates system metrics"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_history = deque(maxlen=10000)
        self.aggregated_metrics = defaultdict(list)
        self._lock = threading.Lock()
        self._collection_thread = None
        self._stop_event = threading.Event()
        
        # Initialize system info
        SYSTEM_INFO.info({
            'app_version': config.app_version,
            'python_version': psutil.PYTHON_VERSION,
            'platform': psutil.SYSTEM,
        })
        
        if config.enable_metrics:
            self.start_collection()
    
    def start_collection(self):
        """Start background metrics collection"""
        if self._collection_thread and self._collection_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._collection_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collection_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop background metrics collection"""
        self._stop_event.set()
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def _collect_loop(self):
        """Background collection loop"""
        while not self._stop_event.is_set():
            try:
                self.collect_system_metrics()
                time.sleep(30)  # Collect every 30 seconds
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                time.sleep(60)  # Back off on error
    
    def collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system.cpu.percent", cpu_percent, "%", {"type": "system"})
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system.memory.percent", memory.percent, "%", {"type": "system"})
            self.record_metric("system.memory.used", memory.used / (1024**3), "GB", {"type": "system"})
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric("system.disk.percent", disk.percent, "%", {"type": "system"})
            
            # Process metrics
            process = psutil.Process()
            self.record_metric("process.cpu.percent", process.cpu_percent(), "%", {"type": "process"})
            self.record_metric("process.memory.mb", process.memory_info().rss / (1024**2), "MB", {"type": "process"})
            self.record_metric("process.threads", process.num_threads(), "count", {"type": "process"})
            
            # Network metrics (if available)
            try:
                net_io = psutil.net_io_counters()
                self.record_metric("network.bytes.sent", net_io.bytes_sent, "bytes", {"type": "network"})
                self.record_metric("network.bytes.recv", net_io.bytes_recv, "bytes", {"type": "network"})
            except:
                pass
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
    
    def record_metric(self, name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None):
        """Record a single metric"""
        metric = PerformanceMetric(
            timestamp=time.time(),
            metric_name=name,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        
        with self._lock:
            self.metrics_history.append(metric)
            self.aggregated_metrics[name].append((metric.timestamp, value))
            
            # Keep only recent values for aggregation
            cutoff_time = time.time() - 3600  # 1 hour
            self.aggregated_metrics[name] = [
                (ts, val) for ts, val in self.aggregated_metrics[name]
                if ts > cutoff_time
            ]
    
    def record_request(self, endpoint: str, duration: float, status: str = "success"):
        """Record API request metrics"""
        REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
        
        self.record_metric(f"request.{endpoint}.duration", duration, "seconds", 
                          {"endpoint": endpoint, "status": status})
    
    def record_inference(self, duration: float, tokens: int, success: bool = True):
        """Record inference metrics"""
        INFERENCE_TIME.observe(duration)
        
        status = "success" if success else "failed"
        self.record_metric("inference.duration", duration, "seconds", {"status": status})
        self.record_metric("inference.tokens", tokens, "count", {"status": status})
    
    def record_document_processed(self, doc_type: str, success: bool = True):
        """Record document processing metrics"""
        status = "success" if success else "failed"
        DOCUMENT_PROCESSED.labels(type=doc_type, status=status).inc()
        
        self.record_metric("document.processed", 1, "count", 
                          {"type": doc_type, "status": status})
    
    def record_cache_access(self, cache_type: str, hit: bool):
        """Record cache access metrics"""
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
            self.record_metric("cache.hit", 1, "count", {"type": cache_type})
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()
            self.record_metric("cache.miss", 1, "count", {"type": cache_type})
    
    def record_error(self, error_type: str, error_message: str):
        """Record error metrics"""
        ERROR_COUNT.labels(error_type=error_type).inc()
        
        logger.error("Application error", 
                    error_type=error_type, 
                    error_message=error_message)
        
        self.record_metric("error", 1, "count", {"type": error_type})
    
    def update_worker_status(self, ready: int, running: int, idle: int):
        """Update worker status gauges"""
        WORKER_COUNT.labels(status="ready").set(ready)
        WORKER_COUNT.labels(status="running").set(running)
        WORKER_COUNT.labels(status="idle").set(idle)
        
        self.record_metric("workers.ready", ready, "count", {"status": "ready"})
        self.record_metric("workers.running", running, "count", {"status": "running"})
        self.record_metric("workers.idle", idle, "count", {"status": "idle"})
    
    def update_cost(self, total_cost: float):
        """Update cost gauge"""
        COST_GAUGE.set(total_cost)
        self.record_metric("cost.total", total_cost, "USD")
    
    def get_metrics_summary(self, period_minutes: int = 60) -> Dict[str, Any]:
        """Get summary of recent metrics"""
        cutoff_time = time.time() - (period_minutes * 60)
        
        with self._lock:
            recent_metrics = [
                m for m in self.metrics_history
                if m.timestamp > cutoff_time
            ]
        
        if not recent_metrics:
            return {"period_minutes": period_minutes, "metrics_count": 0}
        
        # Aggregate by metric name
        summary = {
            "period_minutes": period_minutes,
            "metrics_count": len(recent_metrics),
            "start_time": datetime.fromtimestamp(recent_metrics[0].timestamp).isoformat(),
            "end_time": datetime.fromtimestamp(recent_metrics[-1].timestamp).isoformat(),
            "metrics": {}
        }
        
        # Group metrics
        metric_groups = defaultdict(list)
        for metric in recent_metrics:
            metric_groups[metric.metric_name].append(metric.value)
        
        # Calculate statistics
        for name, values in metric_groups.items():
            if values:
                summary["metrics"][name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": values[-1]
                }
        
        return summary
    
    def get_prometheus_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest()
    
    def create_health_check(self) -> Dict[str, Any]:
        """Create comprehensive health check"""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # System resources check
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        health["checks"]["system_resources"] = {
            "status": "pass" if cpu_percent < 90 and memory_percent < 90 else "warn",
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent
        }
        
        # Process health
        try:
            process = psutil.Process()
            health["checks"]["process"] = {
                "status": "pass",
                "uptime_seconds": time.time() - process.create_time(),
                "memory_mb": process.memory_info().rss / (1024**2),
                "threads": process.num_threads()
            }
        except Exception as e:
            health["checks"]["process"] = {
                "status": "fail",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        # Recent errors check
        recent_errors = self._get_recent_errors()
        health["checks"]["errors"] = {
            "status": "pass" if len(recent_errors) < 10 else "warn",
            "recent_error_count": len(recent_errors)
        }
        
        # Overall status
        if any(check["status"] == "fail" for check in health["checks"].values()):
            health["status"] = "unhealthy"
        elif any(check["status"] == "warn" for check in health["checks"].values()):
            health["status"] = "degraded"
        
        return health
    
    def _get_recent_errors(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get recent errors from metrics"""
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            errors = [
                asdict(m) for m in self.metrics_history
                if m.metric_name == "error" and m.timestamp > cutoff_time
            ]
        
        return errors

class RequestTracker:
    """Track individual request performance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.active_requests = {}
        self._lock = threading.Lock()
    
    def start_request(self, request_id: str, endpoint: str) -> Dict[str, Any]:
        """Start tracking a request"""
        with self._lock:
            self.active_requests[request_id] = {
                "id": request_id,
                "endpoint": endpoint,
                "start_time": time.time(),
                "checkpoints": []
            }
            ACTIVE_REQUESTS.inc()
        
        return self.active_requests[request_id]
    
    def add_checkpoint(self, request_id: str, checkpoint_name: str):
        """Add a checkpoint to request tracking"""
        with self._lock:
            if request_id in self.active_requests:
                self.active_requests[request_id]["checkpoints"].append({
                    "name": checkpoint_name,
                    "timestamp": time.time()
                })
    
    def end_request(self, request_id: str, status: str = "success") -> Optional[Dict[str, Any]]:
        """End request tracking"""
        with self._lock:
            if request_id not in self.active_requests:
                return None
            
            request_data = self.active_requests.pop(request_id)
            ACTIVE_REQUESTS.dec()
        
        # Calculate duration
        end_time = time.time()
        duration = end_time - request_data["start_time"]
        
        # Record metrics
        self.metrics_collector.record_request(
            request_data["endpoint"],
            duration,
            status
        )
        
        # Return complete request data
        request_data["end_time"] = end_time
        request_data["duration"] = duration
        request_data["status"] = status
        
        return request_data

def create_monitoring_middleware(metrics_collector: MetricsCollector):
    """Create monitoring middleware for Streamlit"""
    request_tracker = RequestTracker(metrics_collector)
    
    def middleware(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            request_id = f"req_{int(time.time() * 1000)}"
            endpoint = func.__name__
            
            # Start tracking
            request_tracker.start_request(request_id, endpoint)
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # End tracking - success
                request_tracker.end_request(request_id, "success")
                
                return result
                
            except Exception as e:
                # End tracking - failure
                request_tracker.end_request(request_id, "error")
                
                # Record error
                metrics_collector.record_error(
                    type(e).__name__,
                    str(e)
                )
                
                raise
        
        return wrapper
    
    return middleware
