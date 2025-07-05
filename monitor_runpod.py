#!/usr/bin/env python3
"""
RunPod Monitoring Tool for SmolVLM-GeoEye
=========================================

This script monitors RunPod endpoints, tracks performance metrics,
and provides real-time status updates for the geotechnical workflow.

Author: SmolVLM-GeoEye Team
Version: 1.0.0
"""

import os
import sys
import json
import time
import requests
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import deque
import statistics
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class EndpointMetrics:
    """Metrics for endpoint monitoring"""
    timestamp: str
    status: str
    ready_workers: int
    running_workers: int
    pending_jobs: int
    completed_jobs: int
    failed_jobs: int
    response_time_ms: float
    error: Optional[str] = None

@dataclass
class PerformanceStats:
    """Performance statistics"""
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    uptime_percentage: float
    total_jobs: int
    success_rate: float
    avg_ready_workers: float

class RunPodMonitor:
    """Monitor RunPod endpoints and collect metrics"""
    
    def __init__(self, interval: int = 60, history_size: int = 100):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        self.interval = interval
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.start_time = datetime.now()
        
        if not self.api_key or not self.endpoint_id:
            raise ValueError("RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set")
    
    def collect_metrics(self) -> EndpointMetrics:
        """Collect current endpoint metrics"""
        try:
            start_time = time.time()
            
            # Get endpoint health
            health_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/health"
            health_response = requests.get(
                health_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                workers = health_data.get("workers", {})
                jobs = health_data.get("jobs", {})
                
                metrics = EndpointMetrics(
                    timestamp=datetime.now().isoformat(),
                    status="healthy",
                    ready_workers=workers.get("ready", 0),
                    running_workers=workers.get("running", 0),
                    pending_jobs=jobs.get("pending", 0),
                    completed_jobs=jobs.get("completed", 0),
                    failed_jobs=jobs.get("failed", 0),
                    response_time_ms=response_time_ms
                )
            else:
                metrics = EndpointMetrics(
                    timestamp=datetime.now().isoformat(),
                    status="unhealthy",
                    ready_workers=0,
                    running_workers=0,
                    pending_jobs=0,
                    completed_jobs=0,
                    failed_jobs=0,
                    response_time_ms=response_time_ms,
                    error=f"HTTP {health_response.status_code}"
                )
            
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            metrics = EndpointMetrics(
                timestamp=datetime.now().isoformat(),
                status="error",
                ready_workers=0,
                running_workers=0,
                pending_jobs=0,
                completed_jobs=0,
                failed_jobs=0,
                response_time_ms=0,
                error=str(e)
            )
            self.metrics_history.append(metrics)
            return metrics
    
    def calculate_stats(self) -> PerformanceStats:
        """Calculate performance statistics from metrics history"""
        if not self.metrics_history:
            return PerformanceStats(0, 0, 0, 0, 0, 0, 0)
        
        response_times = [m.response_time_ms for m in self.metrics_history if m.response_time_ms > 0]
        healthy_count = sum(1 for m in self.metrics_history if m.status == "healthy")
        total_count = len(self.metrics_history)
        
        total_completed = max((m.completed_jobs for m in self.metrics_history), default=0)
        total_failed = max((m.failed_jobs for m in self.metrics_history), default=0)
        total_jobs = total_completed + total_failed
        
        ready_workers = [m.ready_workers for m in self.metrics_history]
        
        return PerformanceStats(
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            uptime_percentage=(healthy_count / total_count * 100) if total_count > 0 else 0,
            total_jobs=total_jobs,
            success_rate=(total_completed / total_jobs * 100) if total_jobs > 0 else 0,
            avg_ready_workers=statistics.mean(ready_workers) if ready_workers else 0
        )
    
    def print_status(self, metrics: EndpointMetrics, stats: PerformanceStats):
        """Print formatted status update"""
        # Clear screen (works on Unix-like systems)
        if os.name == 'posix':
            os.system('clear')
        elif os.name == 'nt':
            os.system('cls')
        
        print("="*70)
        print("🏗️  SmolVLM-GeoEye RunPod Endpoint Monitor")
        print("="*70)
        print(f"Endpoint ID: {self.endpoint_id}")
        print(f"Monitoring Duration: {datetime.now() - self.start_time}")
        print(f"Update Interval: {self.interval}s")
        print(f"Last Update: {metrics.timestamp}")
        print("-"*70)
        
        # Current Status
        status_emoji = "✅" if metrics.status == "healthy" else "❌"
        print(f"\n📊 Current Status: {status_emoji} {metrics.status.upper()}")
        
        if metrics.error:
            print(f"   Error: {metrics.error}")
        
        print(f"\n👷 Workers:")
        print(f"   Ready: {metrics.ready_workers}")
        print(f"   Running: {metrics.running_workers}")
        
        print(f"\n📋 Jobs:")
        print(f"   Pending: {metrics.pending_jobs}")
        print(f"   Completed: {metrics.completed_jobs}")
        print(f"   Failed: {metrics.failed_jobs}")
        
        print(f"\n⚡ Response Time: {metrics.response_time_ms:.2f}ms")
        
        # Performance Statistics
        print("\n" + "-"*70)
        print("📈 Performance Statistics (Last {} samples)".format(len(self.metrics_history)))
        print("-"*70)
        
        print(f"\n⏱️  Response Times:")
        print(f"   Average: {stats.avg_response_time:.2f}ms")
        print(f"   Min: {stats.min_response_time:.2f}ms")
        print(f"   Max: {stats.max_response_time:.2f}ms")
        
        print(f"\n📊 Reliability:")
        print(f"   Uptime: {stats.uptime_percentage:.1f}%")
        print(f"   Success Rate: {stats.success_rate:.1f}%")
        print(f"   Total Jobs: {stats.total_jobs}")
        
        print(f"\n👥 Average Ready Workers: {stats.avg_ready_workers:.1f}")
        
        # Cost Estimation (based on RTX 4090 pricing)
        if stats.total_jobs > 0:
            # Rough estimate: $0.0003-0.0013 per request
            min_cost = stats.total_jobs * 0.0003
            max_cost = stats.total_jobs * 0.0013
            print(f"\n💰 Estimated Cost Range: ${min_cost:.4f} - ${max_cost:.4f}")
        
        print("\n" + "="*70)
        print("Press Ctrl+C to stop monitoring")
    
    def save_metrics(self, filename: str = "runpod_metrics.json"):
        """Save metrics history to file"""
        metrics_data = {
            "endpoint_id": self.endpoint_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "metrics": [asdict(m) for m in self.metrics_history],
            "statistics": asdict(self.calculate_stats())
        }
        
        with open(filename, "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Metrics saved to {filename}")
    
    def monitor_once(self) -> Tuple[EndpointMetrics, PerformanceStats]:
        """Run monitoring once and return results"""
        metrics = self.collect_metrics()
        stats = self.calculate_stats()
        return metrics, stats
    
    def monitor_continuous(self):
        """Run continuous monitoring"""
        logger.info(f"Starting continuous monitoring (interval: {self.interval}s)")
        
        try:
            while True:
                metrics = self.collect_metrics()
                stats = self.calculate_stats()
                self.print_status(metrics, stats)
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            self.save_metrics()
            print("\n\n✅ Monitoring stopped. Metrics saved to runpod_metrics.json")
    
    def check_alerts(self, metrics: EndpointMetrics) -> List[str]:
        """Check for alert conditions"""
        alerts = []
        
        # Check worker availability
        if metrics.ready_workers == 0 and metrics.status == "healthy":
            alerts.append("⚠️ No ready workers available!")
        
        # Check job failures
        if metrics.failed_jobs > metrics.completed_jobs * 0.1:  # >10% failure rate
            alerts.append("⚠️ High job failure rate detected!")
        
        # Check response time
        if metrics.response_time_ms > 5000:  # >5 seconds
            alerts.append("⚠️ High response time detected!")
        
        # Check pending jobs
        if metrics.pending_jobs > 10:
            alerts.append("⚠️ High number of pending jobs!")
        
        return alerts

def main():
    """Main monitoring function"""
    parser = argparse.ArgumentParser(
        description="Monitor RunPod endpoints for SmolVLM-GeoEye"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Monitoring interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run monitoring once and exit"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save metrics to specified file"
    )
    parser.add_argument(
        "--history",
        type=int,
        default=100,
        help="Number of samples to keep in history (default: 100)"
    )
    
    args = parser.parse_args()
    
    try:
        monitor = RunPodMonitor(
            interval=args.interval,
            history_size=args.history
        )
        
        if args.once:
            # Run once
            metrics, stats = monitor.monitor_once()
            monitor.print_status(metrics, stats)
            
            # Check for alerts
            alerts = monitor.check_alerts(metrics)
            if alerts:
                print("\n🚨 ALERTS:")
                for alert in alerts:
                    print(f"   {alert}")
            
            if args.save:
                monitor.save_metrics(args.save)
        else:
            # Run continuous monitoring
            monitor.monitor_continuous()
            
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        print("\n❌ Error: RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set in .env file")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Monitoring error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()