#!/usr/bin/env python3
"""
RunPod Configuration Validator for SmolVLM-GeoEye
================================================

This script validates RunPod configuration and tests the connection
to ensure the system is properly set up for geotechnical analysis.

Author: SmolVLM-GeoEye Team
Version: 1.0.0
"""

import os
import sys
import json
import requests
from typing import Dict, Any, List, Tuple
from datetime import datetime
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

class RunPodValidator:
    """Validates RunPod configuration and connectivity"""
    
    def __init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "overall_status": "pending"
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks"""
        logger.info("Starting RunPod validation...")
        
        # Check environment variables
        self._check_environment_variables()
        
        # Check API key format
        self._check_api_key_format()
        
        # Check endpoint ID format
        self._check_endpoint_id_format()
        
        # Test API connectivity
        self._test_api_connectivity()
        
        # Test endpoint health
        self._test_endpoint_health()
        
        # Determine overall status
        all_passed = all(
            check.get("status") == "passed" 
            for check in self.validation_results["checks"].values()
        )
        self.validation_results["overall_status"] = "passed" if all_passed else "failed"
        
        return self.validation_results
    
    def _check_environment_variables(self):
        """Check if required environment variables are set"""
        check_name = "environment_variables"
        
        try:
            missing_vars = []
            
            if not self.api_key:
                missing_vars.append("RUNPOD_API_KEY")
            
            if not self.endpoint_id:
                missing_vars.append("RUNPOD_ENDPOINT_ID")
            
            if missing_vars:
                self.validation_results["checks"][check_name] = {
                    "status": "failed",
                    "message": f"Missing environment variables: {', '.join(missing_vars)}",
                    "details": {
                        "missing": missing_vars,
                        "recommendation": "Please set these variables in your .env file"
                    }
                }
                logger.error(f"❌ Environment variables check failed: {missing_vars}")
            else:
                self.validation_results["checks"][check_name] = {
                    "status": "passed",
                    "message": "All required environment variables are set",
                    "details": {
                        "RUNPOD_API_KEY": "Set (hidden)",
                        "RUNPOD_ENDPOINT_ID": self.endpoint_id
                    }
                }
                logger.info("✅ Environment variables check passed")
                
        except Exception as e:
            self.validation_results["checks"][check_name] = {
                "status": "error",
                "message": f"Error checking environment variables: {str(e)}"
            }
            logger.error(f"❌ Environment variables check error: {str(e)}")
    
    def _check_api_key_format(self):
        """Validate API key format"""
        check_name = "api_key_format"
        
        if not self.api_key:
            self.validation_results["checks"][check_name] = {
                "status": "skipped",
                "message": "API key not set, skipping format check"
            }
            return
        
        try:
            # Basic format checks
            if len(self.api_key) < 20:
                self.validation_results["checks"][check_name] = {
                    "status": "failed",
                    "message": "API key appears to be too short",
                    "details": {
                        "length": len(self.api_key),
                        "expected_min_length": 20
                    }
                }
                logger.error("❌ API key format check failed: too short")
            elif not self.api_key.replace("-", "").replace("_", "").isalnum():
                self.validation_results["checks"][check_name] = {
                    "status": "failed",
                    "message": "API key contains invalid characters",
                    "details": {
                        "recommendation": "API key should only contain alphanumeric characters, hyphens, and underscores"
                    }
                }
                logger.error("❌ API key format check failed: invalid characters")
            else:
                self.validation_results["checks"][check_name] = {
                    "status": "passed",
                    "message": "API key format appears valid",
                    "details": {
                        "length": len(self.api_key),
                        "format": f"***{self.api_key[-4:]}"
                    }
                }
                logger.info("✅ API key format check passed")
                
        except Exception as e:
            self.validation_results["checks"][check_name] = {
                "status": "error",
                "message": f"Error checking API key format: {str(e)}"
            }
            logger.error(f"❌ API key format check error: {str(e)}")
    
    def _check_endpoint_id_format(self):
        """Validate endpoint ID format"""
        check_name = "endpoint_id_format"
        
        if not self.endpoint_id:
            self.validation_results["checks"][check_name] = {
                "status": "skipped",
                "message": "Endpoint ID not set, skipping format check"
            }
            return
        
        try:
            # Basic format checks for RunPod endpoint ID
            if len(self.endpoint_id) < 10:
                self.validation_results["checks"][check_name] = {
                    "status": "failed",
                    "message": "Endpoint ID appears to be too short",
                    "details": {
                        "length": len(self.endpoint_id),
                        "expected_min_length": 10
                    }
                }
                logger.error("❌ Endpoint ID format check failed: too short")
            else:
                self.validation_results["checks"][check_name] = {
                    "status": "passed",
                    "message": "Endpoint ID format appears valid",
                    "details": {
                        "endpoint_id": self.endpoint_id,
                        "length": len(self.endpoint_id)
                    }
                }
                logger.info("✅ Endpoint ID format check passed")
                
        except Exception as e:
            self.validation_results["checks"][check_name] = {
                "status": "error",
                "message": f"Error checking endpoint ID format: {str(e)}"
            }
            logger.error(f"❌ Endpoint ID format check error: {str(e)}")
    
    def _test_api_connectivity(self):
        """Test basic API connectivity"""
        check_name = "api_connectivity"
        
        if not self.api_key:
            self.validation_results["checks"][check_name] = {
                "status": "skipped",
                "message": "API key not set, skipping connectivity check"
            }
            return
        
        try:
            # Test basic API endpoint
            response = requests.get(
                "https://api.runpod.ai/v2/user",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            
            if response.status_code == 200:
                self.validation_results["checks"][check_name] = {
                    "status": "passed",
                    "message": "Successfully connected to RunPod API",
                    "details": {
                        "status_code": response.status_code,
                        "response_time_ms": int(response.elapsed.total_seconds() * 1000)
                    }
                }
                logger.info("✅ API connectivity check passed")
            elif response.status_code == 401:
                self.validation_results["checks"][check_name] = {
                    "status": "failed",
                    "message": "Authentication failed - invalid API key",
                    "details": {
                        "status_code": response.status_code,
                        "recommendation": "Please check your API key is correct"
                    }
                }
                logger.error("❌ API connectivity check failed: authentication error")
            else:
                self.validation_results["checks"][check_name] = {
                    "status": "failed",
                    "message": f"API connection failed with status {response.status_code}",
                    "details": {
                        "status_code": response.status_code,
                        "response": response.text[:200]
                    }
                }
                logger.error(f"❌ API connectivity check failed: status {response.status_code}")
                
        except requests.exceptions.Timeout:
            self.validation_results["checks"][check_name] = {
                "status": "failed",
                "message": "API connection timed out",
                "details": {
                    "timeout": "10 seconds",
                    "recommendation": "Check your internet connection"
                }
            }
            logger.error("❌ API connectivity check failed: timeout")
        except Exception as e:
            self.validation_results["checks"][check_name] = {
                "status": "error",
                "message": f"Error testing API connectivity: {str(e)}"
            }
            logger.error(f"❌ API connectivity check error: {str(e)}")
    
    def _test_endpoint_health(self):
        """Test endpoint health status"""
        check_name = "endpoint_health"
        
        if not self.api_key or not self.endpoint_id:
            self.validation_results["checks"][check_name] = {
                "status": "skipped",
                "message": "API key or endpoint ID not set, skipping health check"
            }
            return
        
        try:
            # Test endpoint health
            url = f"https://api.runpod.ai/v2/{self.endpoint_id}/health"
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            
            if response.status_code == 200:
                health_data = response.json()
                
                # Check if workers are available
                workers = health_data.get("workers", {})
                ready_workers = workers.get("ready", 0)
                
                if ready_workers > 0:
                    status = "passed"
                    message = f"Endpoint is healthy with {ready_workers} ready workers"
                else:
                    status = "warning"
                    message = "Endpoint is accessible but no workers are ready"
                
                self.validation_results["checks"][check_name] = {
                    "status": status,
                    "message": message,
                    "details": {
                        "endpoint_id": self.endpoint_id,
                        "workers": workers,
                        "response_time_ms": int(response.elapsed.total_seconds() * 1000)
                    }
                }
                logger.info(f"✅ Endpoint health check: {message}")
                
            elif response.status_code == 404:
                self.validation_results["checks"][check_name] = {
                    "status": "failed",
                    "message": "Endpoint not found",
                    "details": {
                        "endpoint_id": self.endpoint_id,
                        "recommendation": "Please check your endpoint ID is correct"
                    }
                }
                logger.error("❌ Endpoint health check failed: endpoint not found")
            else:
                self.validation_results["checks"][check_name] = {
                    "status": "failed",
                    "message": f"Health check failed with status {response.status_code}",
                    "details": {
                        "status_code": response.status_code,
                        "response": response.text[:200]
                    }
                }
                logger.error(f"❌ Endpoint health check failed: status {response.status_code}")
                
        except Exception as e:
            self.validation_results["checks"][check_name] = {
                "status": "error",
                "message": f"Error testing endpoint health: {str(e)}"
            }
            logger.error(f"❌ Endpoint health check error: {str(e)}")
    
    def print_summary(self):
        """Print a formatted summary of validation results"""
        print("\n" + "="*60)
        print("🏗️  SmolVLM-GeoEye RunPod Configuration Validation")
        print("="*60)
        print(f"Timestamp: {self.validation_results['timestamp']}")
        print(f"Overall Status: {self.validation_results['overall_status'].upper()}")
        print("\nValidation Checks:")
        print("-"*60)
        
        for check_name, result in self.validation_results["checks"].items():
            status = result["status"]
            message = result["message"]
            
            # Status emoji
            if status == "passed":
                emoji = "✅"
            elif status == "failed":
                emoji = "❌"
            elif status == "warning":
                emoji = "⚠️"
            elif status == "skipped":
                emoji = "⏭️"
            else:
                emoji = "❓"
            
            print(f"{emoji} {check_name.replace('_', ' ').title()}: {message}")
            
            # Print details if failed or warning
            if status in ["failed", "warning"] and "details" in result:
                for key, value in result["details"].items():
                    print(f"   - {key}: {value}")
        
        print("="*60)
        
        # Print recommendations
        if self.validation_results["overall_status"] == "failed":
            print("\n🔧 Recommendations:")
            print("-"*60)
            
            if "environment_variables" in self.validation_results["checks"]:
                if self.validation_results["checks"]["environment_variables"]["status"] == "failed":
                    print("1. Create a .env file in your project root")
                    print("2. Add the following lines:")
                    print("   RUNPOD_API_KEY=your_runpod_api_key_here")
                    print("   RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id_here")
            
            if "api_connectivity" in self.validation_results["checks"]:
                if self.validation_results["checks"]["api_connectivity"]["status"] == "failed":
                    print("1. Verify your API key is correct")
                    print("2. Check if the API key has proper permissions")
                    print("3. Visit https://runpod.io/console/user/settings to manage API keys")
            
            if "endpoint_health" in self.validation_results["checks"]:
                if self.validation_results["checks"]["endpoint_health"]["status"] == "failed":
                    print("1. Verify your endpoint ID is correct")
                    print("2. Check if the endpoint is deployed and running")
                    print("3. Visit https://runpod.io/console/serverless to manage endpoints")
            
            print("\n📚 For detailed setup instructions, see RUNPOD_DEPLOYMENT.md")
        else:
            print("\n✅ All checks passed! Your RunPod configuration is ready for SmolVLM-GeoEye.")
            print("You can now run the application with: streamlit run app.py")
        
        print("\n")

def main():
    """Main validation function"""
    validator = RunPodValidator()
    
    # Run validation
    results = validator.validate_all()
    
    # Print summary
    validator.print_summary()
    
    # Save results to file
    output_file = "runpod_validation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Validation results saved to {output_file}")
    
    # Exit with appropriate code
    if results["overall_status"] == "passed":
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()