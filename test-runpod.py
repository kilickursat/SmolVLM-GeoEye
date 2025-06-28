#!/usr/bin/env python3
"""
RunPod Integration Test Script
==============================

Comprehensive testing suite for RunPod serverless GPU integration.
"""

import os
import sys
import json
import time
import base64
import argparse
import requests
from typing import Dict, Any, Optional
from PIL import Image
import io

class RunPodTester:
    """Test suite for RunPod integration"""
    
    def __init__(self, api_key: str, endpoint_id: str):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = "https://api.runpod.ai/v2"
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        })
    
    def create_test_image(self) -> str:
        """Create a simple test image and return as base64"""
        # Create a simple test image
        img = Image.new('RGB', (400, 300), color='lightblue')
        
        # Add some text-like elements to make it interesting
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw some geometric shapes to simulate an engineering drawing
        draw.rectangle([50, 50, 150, 150], outline='black', width=3)
        draw.rectangle([200, 100, 350, 200], outline='red', width=2)
        draw.line([50, 200, 350, 200], fill='black', width=2)
        draw.line([100, 50, 100, 250], fill='black', width=2)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def test_connection(self) -> Dict[str, Any]:
        """Test basic connection to RunPod"""
        print("🔍 Testing RunPod connection...")
        
        try:
            url = f"{self.base_url}/{self.endpoint_id}/health"
            response = self.session.get(url, timeout=30)
            
            result = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
            
            if result["success"]:
                result["data"] = response.json()
                print("✅ Connection test passed")
                print(f"   Response time: {result['response_time']:.2f}s")
            else:
                result["error"] = response.text
                print(f"❌ Connection test failed: HTTP {response.status_code}")
            
            return result
            
        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def test_sync_inference(self) -> Dict[str, Any]:
        """Test synchronous inference"""
        print("🧠 Testing synchronous inference...")
        
        try:
            # Create test image
            test_image = self.create_test_image()
            
            # Prepare payload
            payload = {
                "input": {
                    "image_data": test_image,
                    "query": "Describe this engineering diagram. What geometric shapes and elements do you see?",
                    "max_new_tokens": 256,
                    "temperature": 0.3
                }
            }
            
            url = f"{self.base_url}/{self.endpoint_id}/runsync"
            
            start_time = time.time()
            response = self.session.post(url, json=payload, timeout=120)
            total_time = time.time() - start_time
            
            result = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "total_time": total_time
            }
            
            if result["success"]:
                data = response.json()
                result["data"] = data
                
                # Extract response text
                output = data.get("output", {})
                response_text = output.get("response", "No response")
                processing_time = output.get("processing_time", "Unknown")
                
                print("✅ Sync inference test passed")
                print(f"   Total time: {total_time:.2f}s")
                print(f"   Processing time: {processing_time}")
                print(f"   Response preview: {response_text[:100]}...")
                
            else:
                result["error"] = response.text
                print(f"❌ Sync inference test failed: HTTP {response.status_code}")
                print(f"   Error: {response.text}")
            
            return result
            
        except Exception as e:
            print(f"❌ Sync inference test failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def test_async_inference(self) -> Dict[str, Any]:
        """Test asynchronous inference"""
        print("⏳ Testing asynchronous inference...")
        
        try:
            # Create test image
            test_image = self.create_test_image()
            
            # Prepare payload
            payload = {
                "input": {
                    "image_data": test_image,
                    "query": "Analyze this technical drawing and identify key structural elements.",
                    "max_new_tokens": 256,
                    "temperature": 0.3
                }
            }
            
            # Submit async job
            url = f"{self.base_url}/{self.endpoint_id}/run"
            
            start_time = time.time()
            response = self.session.post(url, json=payload, timeout=30)
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Failed to submit job: {response.text}"
                }
            
            job_data = response.json()
            job_id = job_data.get("id")
            
            if not job_id:
                return {
                    "success": False,
                    "error": "No job ID received"
                }
            
            print(f"   Job submitted: {job_id}")
            
            # Poll for completion
            status_url = f"{self.base_url}/{self.endpoint_id}/status/{job_id}"
            max_wait = 120  # 2 minutes
            poll_interval = 5
            
            while max_wait > 0:
                status_response = self.session.get(status_url, timeout=30)
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    job_status = status_data.get("status")
                    
                    print(f"   Status: {job_status}")
                    
                    if job_status == "COMPLETED":
                        total_time = time.time() - start_time
                        output = status_data.get("output", {})
                        response_text = output.get("response", "No response")
                        
                        print("✅ Async inference test passed")
                        print(f"   Total time: {total_time:.2f}s")
                        print(f"   Response preview: {response_text[:100]}...")
                        
                        return {
                            "success": True,
                            "job_id": job_id,
                            "total_time": total_time,
                            "data": status_data
                        }
                    
                    elif job_status == "FAILED":
                        return {
                            "success": False,
                            "error": f"Job failed: {status_data.get('error', 'Unknown error')}"
                        }
                    
                    elif job_status in ["IN_PROGRESS", "IN_QUEUE"]:
                        time.sleep(poll_interval)
                        max_wait -= poll_interval
                        continue
                    
                    else:
                        return {
                            "success": False,
                            "error": f"Unknown job status: {job_status}"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to check status: {status_response.text}"
                    }
            
            return {
                "success": False,
                "error": "Job timed out"
            }
            
        except Exception as e:
            print(f"❌ Async inference test failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def test_performance_benchmark(self, num_requests: int = 3) -> Dict[str, Any]:
        """Run performance benchmark"""
        print(f"🏃 Running performance benchmark ({num_requests} requests)...")
        
        results = []
        test_image = self.create_test_image()
        
        queries = [
            "Describe this engineering diagram.",
            "What geometric shapes are visible in this technical drawing?",
            "Analyze the structural elements shown in this image."
        ]
        
        for i in range(num_requests):
            query = queries[i % len(queries)]
            
            try:
                payload = {
                    "input": {
                        "image_data": test_image,
                        "query": query,
                        "max_new_tokens": 128,
                        "temperature": 0.3
                    }
                }
                
                url = f"{self.base_url}/{self.endpoint_id}/runsync"
                
                start_time = time.time()
                response = self.session.post(url, json=payload, timeout=120)
                total_time = time.time() - start_time
                
                result = {
                    "request_id": i + 1,
                    "success": response.status_code == 200,
                    "total_time": total_time,
                    "query": query
                }
                
                if result["success"]:
                    data = response.json()
                    output = data.get("output", {})
                    result["processing_time"] = output.get("processing_time", "Unknown")
                    result["response_length"] = len(output.get("response", ""))
                else:
                    result["error"] = response.text
                
                results.append(result)
                
                print(f"   Request {i+1}: {'✅' if result['success'] else '❌'} {total_time:.2f}s")
                
                # Small delay between requests
                if i < num_requests - 1:
                    time.sleep(2)
                
            except Exception as e:
                results.append({
                    "request_id": i + 1,
                    "success": False,
                    "error": str(e)
                })
                print(f"   Request {i+1}: ❌ {str(e)}")
        
        # Calculate statistics
        successful_results = [r for r in results if r["success"]]
        
        if successful_results:
            times = [r["total_time"] for r in successful_results]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"📊 Benchmark results:")
            print(f"   Successful requests: {len(successful_results)}/{num_requests}")
            print(f"   Average time: {avg_time:.2f}s")
            print(f"   Min time: {min_time:.2f}s")
            print(f"   Max time: {max_time:.2f}s")
        else:
            print("❌ No successful requests in benchmark")
        
        return {
            "total_requests": num_requests,
            "successful_requests": len(successful_results),
            "results": results,
            "statistics": {
                "avg_time": avg_time if successful_results else 0,
                "min_time": min_time if successful_results else 0,
                "max_time": max_time if successful_results else 0
            } if successful_results else None
        }
    
    def monitor_endpoint(self, duration: int = 60) -> Dict[str, Any]:
        """Monitor endpoint for a specified duration"""
        print(f"👀 Monitoring endpoint for {duration} seconds...")
        
        start_time = time.time()
        checks = []
        
        while time.time() - start_time < duration:
            try:
                url = f"{self.base_url}/{self.endpoint_id}/health"
                check_start = time.time()
                response = self.session.get(url, timeout=10)
                response_time = time.time() - check_start
                
                check_result = {
                    "timestamp": time.time(),
                    "success": response.status_code == 200,
                    "response_time": response_time,
                    "status_code": response.status_code
                }
                
                if check_result["success"]:
                    check_result["data"] = response.json()
                else:
                    check_result["error"] = response.text
                
                checks.append(check_result)
                
                status_icon = "✅" if check_result["success"] else "❌"
                elapsed = time.time() - start_time
                print(f"   {status_icon} [{elapsed:.0f}s] Response: {response_time:.2f}s")
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                checks.append({
                    "timestamp": time.time(),
                    "success": False,
                    "error": str(e)
                })
                print(f"   ❌ [{time.time() - start_time:.0f}s] Error: {str(e)}")
                time.sleep(5)
        
        # Calculate uptime statistics
        successful_checks = [c for c in checks if c["success"]]
        uptime_percentage = (len(successful_checks) / len(checks)) * 100 if checks else 0
        
        if successful_checks:
            response_times = [c["response_time"] for c in successful_checks]
            avg_response_time = sum(response_times) / len(response_times)
        else:
            avg_response_time = 0
        
        print(f"📈 Monitoring summary:")
        print(f"   Total checks: {len(checks)}")
        print(f"   Uptime: {uptime_percentage:.1f}%")
        print(f"   Average response time: {avg_response_time:.2f}s")
        
        return {
            "duration": duration,
            "total_checks": len(checks),
            "successful_checks": len(successful_checks),
            "uptime_percentage": uptime_percentage,
            "avg_response_time": avg_response_time,
            "checks": checks
        }

def load_config() -> Dict[str, str]:
    """Load configuration from environment variables or .env file"""
    # Try to load from .env file
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    return {
        'api_key': os.getenv('RUNPOD_API_KEY', ''),
        'endpoint_id': os.getenv('RUNPOD_ENDPOINT_ID', '')
    }

def main():
    parser = argparse.ArgumentParser(description="Test RunPod integration")
    parser.add_argument("--test-connection", action="store_true", help="Test basic connection")
    parser.add_argument("--test-sync", action="store_true", help="Test synchronous inference")
    parser.add_argument("--test-async", action="store_true", help="Test asynchronous inference")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--monitor", type=int, metavar="SECONDS", help="Monitor endpoint for specified duration")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--api-key", help="RunPod API key")
    parser.add_argument("--endpoint-id", help="RunPod endpoint ID")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    api_key = args.api_key or config['api_key']
    endpoint_id = args.endpoint_id or config['endpoint_id']
    
    if not api_key or not endpoint_id:
        print("❌ RunPod API key and endpoint ID are required")
        print("   Set them in .env file or use --api-key and --endpoint-id arguments")
        return 1
    
    # Initialize tester
    tester = RunPodTester(api_key, endpoint_id)
    
    print("🚀 RunPod Integration Test Suite")
    print("=" * 40)
    print(f"Endpoint: {endpoint_id}")
    print()
    
    # Run tests based on arguments
    if args.all or args.test_connection:
        tester.test_connection()
        print()
    
    if args.all or args.test_sync:
        tester.test_sync_inference()
        print()
    
    if args.all or args.test_async:
        tester.test_async_inference()
        print()
    
    if args.all or args.benchmark:
        tester.test_performance_benchmark()
        print()
    
    if args.monitor:
        tester.monitor_endpoint(args.monitor)
        print()
    
    if not any([args.test_connection, args.test_sync, args.test_async, 
                args.benchmark, args.monitor, args.all]):
        parser.print_help()
        return 1
    
    print("🎉 Testing completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
