#!/usr/bin/env python3
"""
Enhanced RunPod Monitoring Tool for SmolVLM-GeoEye
=================================================

ISSUE RESOLUTION: Addresses RunPod $0.00/s usage monitoring issues
- Real-time worker status tracking
- Cost monitoring and alerts
- SmolVLM usage analytics
- Performance metrics collection
- Automated scaling recommendations

Author: SmolVLM-GeoEye Team (Enhanced)
Version: 2.0.0 - Issue Resolution Update
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
import threading
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class EnhancedEndpointMetrics:
    """Enhanced metrics for comprehensive monitoring"""
    timestamp: str
    status: str
    ready_workers: int
    running_workers: int
    idle_workers: int
    pending_jobs: int
    completed_jobs: int
    failed_jobs: int
    response_time_ms: float
    cost_per_hour: float
    smolvlm_active: bool
    worker_utilization: float
    error: Optional[str] = None

@dataclass
class CostAnalytics:
    """Cost analysis and optimization metrics"""
    hourly_cost: float
    daily_projection: float
    monthly_projection: float
    cost_per_job: float
    efficiency_score: float
    optimization_recommendations: List[str]

class EnhancedRunPodMonitor:
    """Enhanced RunPod monitor with cost tracking and optimization"""
    
    def __init__(self, interval: int = 30, history_size: int = 200):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        self.interval = interval
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.start_time = datetime.now()
        
        # Enhanced tracking
        self.cost_per_minute = 0.0013  # Estimated based on RTX A6000
        self.database_path = "runpod_metrics.db"
        self.init_database()
        
        if not self.api_key or not self.endpoint_id:
            raise ValueError("RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set")
    
    def init_database(self):
        """Initialize SQLite database for persistent metrics"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    status TEXT,
                    ready_workers INTEGER,
                    running_workers INTEGER,
                    idle_workers INTEGER,
                    pending_jobs INTEGER,
                    completed_jobs INTEGER,
                    failed_jobs INTEGER,
                    response_time_ms REAL,
                    cost_per_hour REAL,
                    worker_utilization REAL,
                    smolvlm_active BOOLEAN
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cost_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    event_type TEXT,
                    cost REAL,
                    details TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def collect_enhanced_metrics(self) -> EnhancedEndpointMetrics:
        """Collect comprehensive endpoint metrics with cost analysis"""
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
                
                ready_workers = workers.get("ready", 0)
                running_workers = workers.get("running", 0)
                idle_workers = workers.get("idle", 0)
                total_workers = ready_workers + running_workers + idle_workers
                
                # Calculate costs and utilization
                active_workers = running_workers + ready_workers
                hourly_cost = active_workers * self.cost_per_minute * 60
                utilization = (running_workers / total_workers * 100) if total_workers > 0 else 0
                
                metrics = EnhancedEndpointMetrics(
                    timestamp=datetime.now().isoformat(),
                    status="healthy",
                    ready_workers=ready_workers,
                    running_workers=running_workers,
                    idle_workers=idle_workers,
                    pending_jobs=jobs.get("pending", 0),
                    completed_jobs=jobs.get("completed", 0),
                    failed_jobs=jobs.get("failed", 0),
                    response_time_ms=response_time_ms,
                    cost_per_hour=hourly_cost,
                    smolvlm_active=ready_workers > 0,
                    worker_utilization=utilization
                )
            else:
                metrics = EnhancedEndpointMetrics(
                    timestamp=datetime.now().isoformat(),
                    status="unhealthy",
                    ready_workers=0,
                    running_workers=0,
                    idle_workers=0,
                    pending_jobs=0,
                    completed_jobs=0,
                    failed_jobs=0,
                    response_time_ms=response_time_ms,
                    cost_per_hour=0.0,
                    smolvlm_active=False,
                    worker_utilization=0.0,
                    error=f"HTTP {health_response.status_code}"
                )
            
            # Store in database
            self.store_metrics(metrics)
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            metrics = EnhancedEndpointMetrics(
                timestamp=datetime.now().isoformat(),
                status="error",
                ready_workers=0,
                running_workers=0,
                idle_workers=0,
                pending_jobs=0,
                completed_jobs=0,
                failed_jobs=0,
                response_time_ms=0,
                cost_per_hour=0.0,
                smolvlm_active=False,
                worker_utilization=0.0,
                error=str(e)
            )
            self.metrics_history.append(metrics)
            return metrics
    
    def store_metrics(self, metrics: EnhancedEndpointMetrics):
        """Store metrics in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO metrics (
                    timestamp, status, ready_workers, running_workers, idle_workers,
                    pending_jobs, completed_jobs, failed_jobs, response_time_ms,
                    cost_per_hour, worker_utilization, smolvlm_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.status, metrics.ready_workers,
                metrics.running_workers, metrics.idle_workers, metrics.pending_jobs,
                metrics.completed_jobs, metrics.failed_jobs, metrics.response_time_ms,
                metrics.cost_per_hour, metrics.worker_utilization, metrics.smolvlm_active
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    def calculate_cost_analytics(self) -> CostAnalytics:
        """Calculate comprehensive cost analytics"""
        if not self.metrics_history:
            return CostAnalytics(0, 0, 0, 0, 0, ["No data available"])
        
        # Calculate averages
        hourly_costs = [m.cost_per_hour for m in self.metrics_history if m.cost_per_hour > 0]
        avg_hourly_cost = statistics.mean(hourly_costs) if hourly_costs else 0
        
        utilization_rates = [m.worker_utilization for m in self.metrics_history]
        avg_utilization = statistics.mean(utilization_rates) if utilization_rates else 0
        
        # Calculate projections
        daily_projection = avg_hourly_cost * 24
        monthly_projection = daily_projection * 30
        
        # Calculate efficiency
        completed_jobs = max((m.completed_jobs for m in self.metrics_history), default=0)
        cost_per_job = (avg_hourly_cost / max(completed_jobs, 1)) if completed_jobs > 0 else 0
        
        # Efficiency score (0-100)
        efficiency_score = min(avg_utilization, 100)
        
        # Generate recommendations
        recommendations = []
        if avg_utilization < 20:
            recommendations.append("🔻 Low utilization detected - consider reducing worker count")
        elif avg_utilization > 80:
            recommendations.append("🔺 High utilization - consider scaling up for peak demand")
        
        if avg_hourly_cost > 5.0:
            recommendations.append("💰 High costs detected - review worker scaling strategy")
        
        if not any(m.smolvlm_active for m in self.metrics_history[-10:]):
            recommendations.append("⚠️ SmolVLM inactive - check endpoint configuration")
        
        return CostAnalytics(
            hourly_cost=avg_hourly_cost,
            daily_projection=daily_projection,
            monthly_projection=monthly_projection,
            cost_per_job=cost_per_job,
            efficiency_score=efficiency_score,
            optimization_recommendations=recommendations
        )
    
    def print_enhanced_status(self, metrics: EnhancedEndpointMetrics, cost_analytics: CostAnalytics):
        """Print comprehensive status with cost analysis"""
        # Clear screen
        if os.name == 'posix':
            os.system('clear')
        elif os.name == 'nt':
            os.system('cls')
        
        print("=" * 80)
        print("🚀 SmolVLM-GeoEye Enhanced RunPod Monitor")
        print("=" * 80)
        print(f"Endpoint ID: {self.endpoint_id}")
        print(f"Monitoring Duration: {datetime.now() - self.start_time}")
        print(f"Update Interval: {self.interval}s")
        print(f"Last Update: {metrics.timestamp}")
        print("-" * 80)
        
        # Current Status with enhanced indicators
        status_emoji = "✅" if metrics.status == "healthy" else "❌"
        smolvlm_emoji = "🤖" if metrics.smolvlm_active else "🔴"
        
        print(f"\n📊 SYSTEM STATUS: {status_emoji} {metrics.status.upper()}")
        print(f"🤖 SmolVLM Status: {smolvlm_emoji} {'ACTIVE' if metrics.smolvlm_active else 'INACTIVE'}")
        
        if metrics.error:
            print(f"   Error: {metrics.error}")
        
        # Worker Status
        print(f"\n👷 WORKER STATUS:")
        print(f"   Ready: {metrics.ready_workers} workers")
        print(f"   Running: {metrics.running_workers} workers")
        print(f"   Idle: {metrics.idle_workers} workers")
        print(f"   Utilization: {metrics.worker_utilization:.1f}%")
        
        # Job Queue
        print(f"\n📋 JOB QUEUE:")
        print(f"   Pending: {metrics.pending_jobs}")
        print(f"   Completed: {metrics.completed_jobs}")
        print(f"   Failed: {metrics.failed_jobs}")
        
        # Performance
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Response Time: {metrics.response_time_ms:.2f}ms")
        success_rate = (metrics.completed_jobs / max(metrics.completed_jobs + metrics.failed_jobs, 1)) * 100
        print(f"   Success Rate: {success_rate:.1f}%")
        
        # Cost Analysis
        print("\n" + "-" * 80)
        print("💰 COST ANALYSIS")
        print("-" * 80)
        
        print(f"\n💸 CURRENT COSTS:")
        print(f"   Hourly Rate: ${cost_analytics.hourly_cost:.4f}/hour")
        print(f"   Daily Projection: ${cost_analytics.daily_projection:.2f}/day")
        print(f"   Monthly Projection: ${cost_analytics.monthly_projection:.2f}/month")
        
        if cost_analytics.cost_per_job > 0:
            print(f"   Cost per Job: ${cost_analytics.cost_per_job:.4f}")
        
        print(f"\n📈 EFFICIENCY:")
        print(f"   Efficiency Score: {cost_analytics.efficiency_score:.1f}/100")
        
        # Optimization Recommendations
        if cost_analytics.optimization_recommendations:
            print(f"\n🎯 OPTIMIZATION RECOMMENDATIONS:")
            for rec in cost_analytics.optimization_recommendations:
                print(f"   {rec}")
        
        # Historical Performance
        if len(self.metrics_history) >= 5:
            print("\n" + "-" * 80)
            print("📈 HISTORICAL PERFORMANCE (Last 10 samples)")
            print("-" * 80)
            
            recent_metrics = list(self.metrics_history)[-10:]
            avg_workers = statistics.mean([m.ready_workers + m.running_workers for m in recent_metrics])
            avg_response = statistics.mean([m.response_time_ms for m in recent_metrics if m.response_time_ms > 0])
            avg_cost = statistics.mean([m.cost_per_hour for m in recent_metrics])
            
            print(f"\n📊 AVERAGES:")
            print(f"   Workers: {avg_workers:.1f}")
            print(f"   Response Time: {avg_response:.1f}ms")
            print(f"   Hourly Cost: ${avg_cost:.4f}")
        
        print("\n" + "=" * 80)
        print("Press Ctrl+C to stop monitoring | 🔄 Auto-refresh every {}s".format(self.interval))
    
    def generate_cost_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate detailed cost report"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get data from last N hours
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            cursor.execute('''
                SELECT * FROM metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {"error": "No data available for the specified period"}
            
            # Calculate report metrics
            total_cost = sum(row[10] for row in rows) / len(rows) * hours  # cost_per_hour * hours
            avg_utilization = statistics.mean([row[11] for row in rows])  # worker_utilization
            peak_workers = max(row[3] + row[4] for row in rows)  # running + ready workers
            
            return {
                "period_hours": hours,
                "total_cost": total_cost,
                "average_utilization": avg_utilization,
                "peak_workers": peak_workers,
                "data_points": len(rows),
                "cost_efficiency": avg_utilization / max(total_cost, 0.01),
                "recommendations": self.calculate_cost_analytics().optimization_recommendations
            }
            
        except Exception as e:
            return {"error": f"Failed to generate report: {e}"}
    
    def monitor_continuous_enhanced(self):
        """Run enhanced continuous monitoring"""
        logger.info(f"Starting enhanced monitoring (interval: {self.interval}s)")
        
        try:
            while True:
                metrics = self.collect_enhanced_metrics()
                cost_analytics = self.calculate_cost_analytics()
                self.print_enhanced_status(metrics, cost_analytics)
                
                # Check for alerts
                self.check_enhanced_alerts(metrics, cost_analytics)
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            self.save_enhanced_report()
            print("\n\n✅ Enhanced monitoring stopped. Report saved.")
    
    def check_enhanced_alerts(self, metrics: EnhancedEndpointMetrics, cost_analytics: CostAnalytics):
        """Check for enhanced alert conditions"""
        alerts = []
        
        # Worker alerts
        if metrics.ready_workers == 0 and metrics.status == "healthy":
            alerts.append("⚠️ CRITICAL: No ready workers available!")
        
        # Cost alerts
        if cost_analytics.hourly_cost > 10.0:
            alerts.append(f"💰 HIGH COST ALERT: ${cost_analytics.hourly_cost:.2f}/hour")
        
        # Efficiency alerts
        if cost_analytics.efficiency_score < 10:
            alerts.append("📉 LOW EFFICIENCY: Worker utilization below 10%")
        
        # SmolVLM alerts
        if not metrics.smolvlm_active:
            alerts.append("🤖 SmolVLM INACTIVE: Check endpoint configuration")
        
        # Performance alerts
        if metrics.response_time_ms > 10000:
            alerts.append("🐌 SLOW RESPONSE: >10 seconds response time")
        
        if alerts:
            logger.warning("ALERTS DETECTED:")
            for alert in alerts:
                logger.warning(f"  {alert}")
    
    def save_enhanced_report(self):
        """Save enhanced monitoring report"""
        report = {
            "session_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
                "endpoint_id": self.endpoint_id
            },
            "cost_analysis": asdict(self.calculate_cost_analytics()),
            "recent_metrics": [asdict(m) for m in list(self.metrics_history)[-20:]],
            "cost_report_24h": self.generate_cost_report(24),
            "recommendations": self.calculate_cost_analytics().optimization_recommendations
        }
        
        filename = f"enhanced_runpod_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Enhanced report saved to {filename}")

def main():
    """Enhanced main monitoring function"""
    parser = argparse.ArgumentParser(
        description="Enhanced RunPod monitoring for SmolVLM-GeoEye with cost tracking"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Monitoring interval in seconds (default: 30)"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate cost report and exit"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours for cost report (default: 24)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run monitoring once and exit"
    )
    
    args = parser.parse_args()
    
    try:
        monitor = EnhancedRunPodMonitor(interval=args.interval)
        
        if args.report:
            # Generate cost report
            report = monitor.generate_cost_report(args.hours)
            print("\n📊 COST REPORT")
            print("=" * 50)
            print(json.dumps(report, indent=2))
            
        elif args.once:
            # Run once
            metrics = monitor.collect_enhanced_metrics()
            cost_analytics = monitor.calculate_cost_analytics()
            monitor.print_enhanced_status(metrics, cost_analytics)
            
        else:
            # Run continuous monitoring
            monitor.monitor_continuous_enhanced()
            
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        print("\n❌ Error: RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set in .env file")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Monitoring error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
