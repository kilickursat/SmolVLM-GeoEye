#!/usr/bin/env python3
"""
SmolVLM-GeoEye Enhanced RunPod Client Module
===========================================

Enhanced client for SmolVLM inference with cost tracking,
performance monitoring, and auto-scaling capabilities.

Author: SmolVLM-GeoEye Team
Version: 3.1.0
"""

import logging
import time
import json
import requests
import base64
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import aiohttp
from threading import Lock
import statistics

logger = logging.getLogger(__name__)

@dataclass
class InferenceMetrics:
    """Metrics for a single inference"""
    job_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    tokens_generated: int
    cost_estimate: float
    success: bool
    error: Optional[str] = None

class EnhancedRunPodClient:
    """Enhanced RunPod client with monitoring and optimization"""
    
    def __init__(self, config):
        self.config = config
        self.base_url = f"https://api.runpod.ai/v2/{config.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        # Metrics tracking
        self.metrics_history = []
        self.metrics_lock = Lock()
        self.total_cost = 0.0
        self.total_jobs = 0
        self.success_count = 0
        
        # Performance tracking
        self.response_times = []
        self.worker_status = {
            'ready': 0,
            'running': 0,
            'idle': 0,
            'last_check': None
        }
        
        # Cost tracking
        self.cost_per_minute = config.cost_per_minute
        self.cost_per_token = 0.00001  # Estimated
        
    def health_check(self) -> Dict[str, Any]:
        """Check endpoint health"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=self.config.health_check_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self.worker_status.update({
                    'ready': data.get('workers', {}).get('ready', 0),
                    'running': data.get('workers', {}).get('running', 0),
                    'idle': data.get('workers', {}).get('idle', 0),
                    'last_check': datetime.now().isoformat()
                })
                
                return {
                    "status": "healthy",
                    "ready": True,
                    "workers": self.worker_status,
                    "jobs": data.get('jobs', {})
                }
            else:
                return {
                    "status": "unhealthy",
                    "ready": False,
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "ready": False,
                "error": str(e)
            }
    
    def enhanced_health_check(self) -> Dict[str, Any]:
        """Enhanced health check with cost and performance metrics"""
        basic_health = self.health_check()
        
        # Add enhanced metrics
        with self.metrics_lock:
            avg_response_time = statistics.mean(self.response_times[-10:]) if self.response_times else 0
            success_rate = (self.success_count / self.total_jobs * 100) if self.total_jobs > 0 else 0
            
            enhanced = {
                **basic_health,
                'metrics': {
                    'total_jobs': self.total_jobs,
                    'success_rate': success_rate,
                    'avg_response_time_ms': avg_response_time * 1000,
                    'total_cost': self.total_cost,
                    'cost_per_job': self.total_cost / max(self.total_jobs, 1)
                },
                'cost_per_hour': self._calculate_hourly_cost(),
                'recommendations': self._get_optimization_recommendations()
            }
        
        return enhanced
    
    def run_sync(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run synchronous inference"""
        job_id = f"job_{int(time.time() * 1000)}"
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/runsync",
                headers=self.headers,
                json={"input": input_data},
                timeout=self.config.runpod_timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                output = result.get("output", {})
                
                # Track metrics
                tokens = len(output.get("response", "").split()) * 1.5  # Rough estimate
                cost = self._calculate_job_cost(duration, tokens)
                
                metrics = InferenceMetrics(
                    job_id=job_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                    tokens_generated=int(tokens),
                    cost_estimate=cost,
                    success=True
                )
                
                self._record_metrics(metrics)
                
                return {
                    "status": "success",
                    "output": output,
                    "metrics": asdict(metrics),
                    "response": output.get("response", "")
                }
            else:
                error_msg = f"RunPod error: {response.status_code} - {response.text}"
                
                metrics = InferenceMetrics(
                    job_id=job_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                    tokens_generated=0,
                    cost_estimate=0,
                    success=False,
                    error=error_msg
                )
                
                self._record_metrics(metrics)
                
                return {
                    "status": "error",
                    "error": error_msg,
                    "metrics": asdict(metrics)
                }
                
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            error_msg = f"Request failed: {str(e)}"
            
            metrics = InferenceMetrics(
                job_id=job_id,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                tokens_generated=0,
                cost_estimate=0,
                success=False,
                error=error_msg
            )
            
            self._record_metrics(metrics)
            
            return {
                "status": "error",
                "error": error_msg,
                "metrics": asdict(metrics)
            }
    
    def run_sync_with_tracking(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run sync with enhanced tracking and retries"""
        retries = 0
        last_error = None
        
        while retries < self.config.runpod_max_retries:
            result = self.run_sync(input_data)
            
            if result["status"] == "success":
                return result
            
            last_error = result.get("error", "Unknown error")
            retries += 1
            
            if retries < self.config.runpod_max_retries:
                wait_time = min(2 ** retries, 30)  # Exponential backoff
                logger.warning(f"Retry {retries}/{self.config.runpod_max_retries} after {wait_time}s")
                time.sleep(wait_time)
        
        # All retries failed
        logger.error(f"All retries failed. Last error: {last_error}")
        return {
            "status": "error",
            "error": f"Failed after {retries} retries. Last error: {last_error}",
            "retries": retries
        }
    
    async def run_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run asynchronous inference"""
        job_id = f"job_{int(time.time() * 1000)}"
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                # Submit job
                async with session.post(
                    f"{self.base_url}/run",
                    headers=self.headers,
                    json={"input": input_data}
                ) as response:
                    if response.status != 200:
                        return {
                            "status": "error",
                            "error": f"Failed to submit job: {response.status}"
                        }
                    
                    job_data = await response.json()
                    job_id = job_data.get("id", job_id)
                
                # Poll for completion
                while True:
                    async with session.get(
                        f"{self.base_url}/status/{job_id}",
                        headers=self.headers
                    ) as response:
                        if response.status != 200:
                            return {
                                "status": "error",
                                "error": f"Failed to check status: {response.status}"
                            }
                        
                        status_data = await response.json()
                        status = status_data.get("status")
                        
                        if status == "COMPLETED":
                            end_time = time.time()
                            duration = end_time - start_time
                            output = status_data.get("output", {})
                            
                            # Track metrics
                            tokens = len(output.get("response", "").split()) * 1.5
                            cost = self._calculate_job_cost(duration, tokens)
                            
                            metrics = InferenceMetrics(
                                job_id=job_id,
                                start_time=start_time,
                                end_time=end_time,
                                duration_seconds=duration,
                                tokens_generated=int(tokens),
                                cost_estimate=cost,
                                success=True
                            )
                            
                            self._record_metrics(metrics)
                            
                            return {
                                "status": "success",
                                "output": output,
                                "metrics": asdict(metrics),
                                "response": output.get("response", "")
                            }
                        
                        elif status == "FAILED":
                            return {
                                "status": "error",
                                "error": status_data.get("error", "Job failed")
                            }
                        
                        # Still processing
                        await asyncio.sleep(2)
                        
                        # Check timeout
                        if time.time() - start_time > self.config.runpod_timeout:
                            return {
                                "status": "error",
                                "error": "Job timed out"
                            }
                            
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Async request failed: {str(e)}"
                }
    
    def batch_process(self, items: List[Dict[str, Any]], max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Process multiple items in batch with concurrency control"""
        results = []
        
        async def process_batch():
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_item(item):
                async with semaphore:
                    return await self.run_async(item)
            
            tasks = [process_item(item) for item in items]
            return await asyncio.gather(*tasks)
        
        # Run async batch processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(process_batch())
        finally:
            loop.close()
        
        return results
    
    def _record_metrics(self, metrics: InferenceMetrics):
        """Record metrics for tracking"""
        with self.metrics_lock:
            self.metrics_history.append(metrics)
            self.total_jobs += 1
            
            if metrics.success:
                self.success_count += 1
                self.total_cost += metrics.cost_estimate
                self.response_times.append(metrics.duration_seconds)
                
                # Keep only recent response times
                if len(self.response_times) > 100:
                    self.response_times = self.response_times[-100:]
            
            # Keep only recent history
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
    
    def _calculate_job_cost(self, duration_seconds: float, tokens: int) -> float:
        """Calculate estimated cost for a job"""
        time_cost = (duration_seconds / 60) * self.cost_per_minute
        token_cost = tokens * self.cost_per_token
        return time_cost + token_cost
    
    def _calculate_hourly_cost(self) -> float:
        """Calculate current hourly cost rate"""
        if not self.worker_status['ready']:
            return 0.0
        
        active_workers = self.worker_status['ready'] + self.worker_status['running']
        return active_workers * self.cost_per_minute * 60
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Get cost optimization recommendations"""
        recommendations = []
        
        # Check utilization
        total_workers = sum([
            self.worker_status['ready'],
            self.worker_status['running'],
            self.worker_status['idle']
        ])
        
        if total_workers > 0:
            utilization = self.worker_status['running'] / total_workers
            
            if utilization < 0.2:
                recommendations.append("Low utilization - consider reducing workers")
            elif utilization > 0.8:
                recommendations.append("High utilization - consider adding workers")
        
        # Check success rate
        if self.total_jobs > 10:
            success_rate = self.success_count / self.total_jobs
            if success_rate < 0.9:
                recommendations.append("Low success rate - check error logs")
        
        # Check response times
        if self.response_times and statistics.mean(self.response_times) > 30:
            recommendations.append("High response times - optimize input size or add workers")
        
        return recommendations
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        with self.metrics_lock:
            if not self.metrics_history:
                return {
                    "total_jobs": 0,
                    "success_rate": 0,
                    "total_cost": 0,
                    "avg_response_time": 0,
                    "cost_per_job": 0
                }
            
            recent_metrics = self.metrics_history[-100:]
            successful_jobs = [m for m in recent_metrics if m.success]
            
            return {
                "total_jobs": self.total_jobs,
                "success_count": self.success_count,
                "success_rate": (self.success_count / self.total_jobs * 100) if self.total_jobs > 0 else 0,
                "total_cost": self.total_cost,
                "avg_response_time": statistics.mean([m.duration_seconds for m in successful_jobs]) if successful_jobs else 0,
                "avg_tokens": statistics.mean([m.tokens_generated for m in successful_jobs]) if successful_jobs else 0,
                "cost_per_job": self.total_cost / max(self.total_jobs, 1),
                "hourly_cost": self._calculate_hourly_cost(),
                "worker_status": self.worker_status,
                "recent_errors": [m.error for m in recent_metrics if not m.success and m.error][-5:]
            }
    
    def scale_workers(self, target_count: int) -> Dict[str, Any]:
        """Scale workers to target count"""
        try:
            response = requests.patch(
                f"{self.base_url}/workers",
                headers=self.headers,
                json={"workers": target_count}
            )
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "workers": target_count,
                    "message": f"Scaled to {target_count} workers"
                }
            else:
                return {
                    "status": "error",
                    "error": f"Failed to scale: {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Scaling failed: {str(e)}"
            }
    
    def auto_scale(self) -> Optional[Dict[str, Any]]:
        """Auto-scale based on current metrics"""
        health = self.health_check()
        
        if health["status"] != "healthy":
            return None
        
        total_workers = sum([
            self.worker_status['ready'],
            self.worker_status['running'],
            self.worker_status['idle']
        ])
        
        if total_workers == 0:
            return None
        
        utilization = self.worker_status['running'] / total_workers
        
        # Scale up if high utilization
        if utilization > self.config.scale_up_threshold and total_workers < self.config.max_workers:
            new_count = min(total_workers + 1, self.config.max_workers)
            return self.scale_workers(new_count)
        
        # Scale down if low utilization
        elif utilization < self.config.scale_down_threshold and total_workers > self.config.min_workers:
            new_count = max(total_workers - 1, self.config.min_workers)
            return self.scale_workers(new_count)
        
        return None
