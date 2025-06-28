#!/bin/bash

# setup.sh - Enhanced Streamlit Configuration for RunPod Integration
# ==================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up Geotechnical Engineering Workflow with RunPod Integration${NC}"

# Create Streamlit configuration directory
mkdir -p ~/.streamlit/

# Create Streamlit configuration file
echo -e "${YELLOW}📝 Creating Streamlit configuration...${NC}"
cat > ~/.streamlit/config.toml << EOF
[general]
email = ""

[server]
headless = true
enableCORS = false
port = \$PORT

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#262730"

[browser]
serverAddress = "0.0.0.0"
serverPort = \$PORT
gatherUsageStats = false

[logger]
level = "info"

[client]
showErrorDetails = false
EOF

# Create enhanced credentials file for Streamlit secrets
echo -e "${YELLOW}🔐 Creating Streamlit secrets template...${NC}"
cat > ~/.streamlit/secrets.toml << EOF
# Streamlit Secrets Configuration for RunPod Integration
# ======================================================

# RunPod Configuration (REQUIRED for GPU processing)
# Get these from: https://runpod.io/console/user/settings
# RUNPOD_API_KEY = "your_runpod_api_key_here"
# RUNPOD_ENDPOINT_ID = "your_runpod_endpoint_id_here"

# HuggingFace Token (Optional but recommended for model access)
# Get your token from: https://huggingface.co/settings/tokens
# HF_TOKEN = "hf_your_token_here"

# Environment Configuration
ENVIRONMENT = "production"
DEBUG = false

# Application Settings
MAX_UPLOAD_SIZE = 200
TIMEOUT_SECONDS = 300

# Example RunPod Configuration (uncomment and fill in your values):
# [runpod]
# api_key = "your_runpod_api_key"
# endpoint_id = "your_runpod_endpoint_id"
# base_url = "https://api.runpod.ai/v2"
# timeout = 300
# max_retries = 3
EOF

# Create environment file template
echo -e "${YELLOW}🌍 Creating environment file template...${NC}"
cat > .env.example << EOF
# Environment Variables for Geotechnical Engineering Workflow
# ==========================================================

# RunPod Configuration (Primary GPU Processing)
RUNPOD_API_KEY=your_runpod_api_key_here
RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id_here

# HuggingFace Configuration (Optional)
HF_TOKEN=hf_your_token_here

# Application Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Processing Configuration
MAX_UPLOAD_SIZE_MB=200
PROCESSING_TIMEOUT=300
ENABLE_ASYNC_PROCESSING=true

# Security Configuration (for production)
# SSL_CERT_PATH=/path/to/cert.pem
# SSL_KEY_PATH=/path/to/key.pem
EOF

# Create local environment file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}📋 Creating local environment file...${NC}"
    cp .env.example .env
    echo -e "${RED}⚠️  Please edit .env file with your actual RunPod credentials${NC}"
fi

# Create RunPod configuration validation script
echo -e "${YELLOW}🔧 Creating RunPod configuration validator...${NC}"
cat > validate_runpod.py << 'EOF'
#!/usr/bin/env python3
"""
RunPod Configuration Validator
=============================

Validates RunPod configuration and tests connectivity.
"""

import os
import sys
import requests
import json
from typing import Dict, Any

def load_config() -> Dict[str, str]:
    """Load configuration from environment variables or .env file"""
    config = {}
    
    # Try to load from .env file
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    # Get required configuration
    config['api_key'] = os.getenv('RUNPOD_API_KEY', '')
    config['endpoint_id'] = os.getenv('RUNPOD_ENDPOINT_ID', '')
    
    return config

def validate_api_key(api_key: str) -> bool:
    """Validate API key format"""
    return api_key.startswith('runpod-') and len(api_key) > 20

def test_endpoint_health(api_key: str, endpoint_id: str) -> Dict[str, Any]:
    """Test RunPod endpoint health"""
    try:
        url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    print("🚀 RunPod Configuration Validator")
    print("=" * 40)
    
    # Load configuration
    config = load_config()
    
    # Check API key
    api_key = config['api_key']
    if not api_key:
        print("❌ RUNPOD_API_KEY not found")
        print("   Please set it in .env file or environment variables")
        return False
    
    if not validate_api_key(api_key):
        print("⚠️  API key format may be invalid (should start with 'runpod-')")
    else:
        print(f"✅ API key found: {api_key[:15]}...")
    
    # Check endpoint ID
    endpoint_id = config['endpoint_id']
    if not endpoint_id:
        print("❌ RUNPOD_ENDPOINT_ID not found")
        print("   Please set it in .env file or environment variables")
        return False
    else:
        print(f"✅ Endpoint ID found: {endpoint_id}")
    
    # Test endpoint health
    print("\n🔍 Testing endpoint connectivity...")
    health_result = test_endpoint_health(api_key, endpoint_id)
    
    if health_result["success"]:
        print("✅ Endpoint is healthy!")
        print(f"   Response: {json.dumps(health_result['response'], indent=2)}")
        return True
    else:
        print(f"❌ Endpoint test failed:")
        if "error" in health_result:
            print(f"   Error: {health_result['error']}")
        else:
            print(f"   Status: {health_result['status_code']}")
            print(f"   Response: {health_result['response']}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

chmod +x validate_runpod.py

# Create quick setup script for development
echo -e "${YELLOW}⚡ Creating quick development setup script...${NC}"
cat > quick_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Quick Setup Script for Development Environment
==============================================

Sets up virtual environment and installs dependencies.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    print("🚀 Quick Development Environment Setup")
    print("=" * 45)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Create virtual environment
    if not os.path.exists("geotechnical_env"):
        if not run_command("python -m venv geotechnical_env", "Creating virtual environment"):
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = "geotechnical_env\\Scripts\\activate"
        pip_cmd = "geotechnical_env\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = "source geotechnical_env/bin/activate"
        pip_cmd = "geotechnical_env/bin/pip"
    
    # Install dependencies
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing main dependencies"):
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':
        print("   geotechnical_env\\Scripts\\activate")
    else:
        print("   source geotechnical_env/bin/activate")
    
    print("2. Configure RunPod credentials in .env file")
    print("3. Validate configuration: python validate_runpod.py")
    print("4. Run the app: streamlit run app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

chmod +x quick_setup.py

# Create deployment checklist
echo -e "${YELLOW}📋 Creating deployment checklist...${NC}"
cat > DEPLOYMENT_CHECKLIST.md << 'EOF'
# 🚀 Deployment Checklist

## Pre-Deployment

### RunPod Setup
- [ ] RunPod account created
- [ ] API key generated
- [ ] Docker image built and pushed
- [ ] RunPod template created
- [ ] Serverless endpoint deployed
- [ ] Endpoint health tested

### Configuration
- [ ] `.env` file configured with RunPod credentials
- [ ] Streamlit secrets configured
- [ ] HuggingFace token added (optional)
- [ ] Configuration validated with `python validate_runpod.py`

### Testing
- [ ] Local development environment tested
- [ ] RunPod endpoint connectivity tested
- [ ] Document processing tested
- [ ] Vision analysis tested
- [ ] Agent responses tested

## Deployment Options

### 1. Streamlit Cloud
- [ ] Repository pushed to GitHub
- [ ] Streamlit Cloud app created
- [ ] Secrets configured in Streamlit Cloud dashboard
- [ ] App deployed and tested

### 2. Heroku
- [ ] Heroku CLI installed
- [ ] Heroku app created
- [ ] Config vars set
- [ ] App deployed via Git

### 3. Google Cloud Run
- [ ] Google Cloud project set up
- [ ] Docker image pushed to Google Container Registry
- [ ] Cloud Run service deployed
- [ ] Environment variables configured

### 4. AWS ECS/Fargate
- [ ] AWS account configured
- [ ] ECR repository created
- [ ] Task definition created
- [ ] Service deployed

### 5. Self-Hosted
- [ ] Server provisioned
- [ ] Docker/Docker Compose installed
- [ ] SSL certificates configured
- [ ] Reverse proxy configured (nginx/caddy)
- [ ] Monitoring set up

## Post-Deployment

### Verification
- [ ] App loads successfully
- [ ] RunPod status shows healthy
- [ ] Document upload works
- [ ] Vision analysis processes correctly
- [ ] Chat interface responds properly
- [ ] Visualizations render correctly

### Monitoring
- [ ] Application logs monitored
- [ ] RunPod usage tracked
- [ ] Error alerts configured
- [ ] Performance metrics collected

### Optimization
- [ ] RunPod endpoint settings optimized for cost/performance
- [ ] Idle timeout configured appropriately
- [ ] Auto-scaling parameters tuned
- [ ] Flash Boot enabled for faster cold starts

## Troubleshooting Common Issues

### RunPod Connection Issues
- Verify API key and endpoint ID
- Check endpoint status in RunPod console
- Test with curl command
- Review endpoint logs

### Streamlit Issues
- Check secrets configuration
- Verify port settings
- Review application logs
- Test locally first

### Performance Issues
- Monitor RunPod GPU utilization
- Adjust timeout settings
- Consider upgrading GPU tier
- Enable Flash Boot

### Cost Optimization
- Set minimum workers to 0
- Configure appropriate idle timeout
- Monitor usage patterns
- Consider reserved capacity for high usage
EOF

# Create monitoring script
echo -e "${YELLOW}📊 Creating monitoring script...${NC}"
cat > monitor_runpod.py << 'EOF'
#!/usr/bin/env python3
"""
RunPod Monitoring Script
========================

Monitors RunPod endpoint status and usage.
"""

import requests
import time
import json
import os
from datetime import datetime
from typing import Dict, Any

def load_config() -> Dict[str, str]:
    """Load RunPod configuration"""
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

def check_endpoint_status(api_key: str, endpoint_id: str) -> Dict[str, Any]:
    """Check endpoint status"""
    try:
        url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": str(e)
        }

def monitor_endpoint(api_key: str, endpoint_id: str, interval: int = 60):
    """Monitor endpoint continuously"""
    print(f"🔍 Starting RunPod endpoint monitoring (interval: {interval}s)")
    print(f"📍 Endpoint: {endpoint_id}")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            status = check_endpoint_status(api_key, endpoint_id)
            
            timestamp = status["timestamp"]
            if status["success"]:
                print(f"✅ [{timestamp}] Endpoint healthy")
                if "response" in status:
                    workers = status["response"].get("workers", {})
                    print(f"   Workers: {workers}")
            else:
                print(f"❌ [{timestamp}] Endpoint issue:")
                if "error" in status:
                    print(f"   Error: {status['error']}")
                else:
                    print(f"   Status: {status['status_code']}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor RunPod endpoint")
    parser.add_argument("--interval", "-i", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    
    args = parser.parse_args()
    
    config = load_config()
    
    if not config['api_key'] or not config['endpoint_id']:
        print("❌ RunPod configuration not found")
        print("   Please set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID")
        return 1
    
    if args.once:
        status = check_endpoint_status(config['api_key'], config['endpoint_id'])
        print(json.dumps(status, indent=2))
    else:
        monitor_endpoint(config['api_key'], config['endpoint_id'], args.interval)
    
    return 0

if __name__ == "__main__":
    exit(main())
EOF

chmod +x monitor_runpod.py

# Create final summary
echo ""
echo -e "${GREEN}✅ Streamlit configuration files created successfully!${NC}"
echo ""
echo -e "${BLUE}📁 Created files:${NC}"
echo "   ~/.streamlit/config.toml       - Streamlit configuration"
echo "   ~/.streamlit/secrets.toml      - Secrets template"
echo "   .env.example                   - Environment variables template"
echo "   .env                          - Local environment file"
echo "   validate_runpod.py            - RunPod configuration validator"
echo "   quick_setup.py                - Development environment setup"
echo "   monitor_runpod.py             - RunPod monitoring script"
echo "   DEPLOYMENT_CHECKLIST.md       - Deployment checklist"
echo ""
echo -e "${YELLOW}🔧 Next steps:${NC}"
echo "1. 📝 Edit .env file with your RunPod credentials"
echo "2. 🔍 Validate configuration: python validate_runpod.py"
echo "3. 🚀 Deploy RunPod worker (see deploy-runpod.sh)"
echo "4. ▶️  Run the app: streamlit run app.py"
echo ""
echo -e "${GREEN}🎉 Ready for RunPod GPU-powered deployment!${NC}"
