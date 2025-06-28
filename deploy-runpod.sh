#!/bin/bash

# RunPod Deployment Script for SmolVLM Geotechnical Engineering Workflow
# =======================================================================

set -e  # Exit on any error

# Configuration
IMAGE_NAME="geotechnical-smolvlm-runpod"
REGISTRY="your-registry"  # Replace with your Docker registry
VERSION="v1.0.0"
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${VERSION}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    if ! command -v curl &> /dev/null; then
        log_error "curl is required but not installed"
        exit 1
    fi
    
    log_success "Dependencies check passed"
}

build_image() {
    log_info "Building Docker image: ${FULL_IMAGE_NAME}"
    
    # Build the Docker image
    docker build -f Dockerfile.runpod -t ${FULL_IMAGE_NAME} .
    
    if [ $? -eq 0 ]; then
        log_success "Docker image built successfully"
    else
        log_error "Docker image build failed"
        exit 1
    fi
}

push_image() {
    log_info "Pushing Docker image to registry..."
    
    # Push to registry
    docker push ${FULL_IMAGE_NAME}
    
    if [ $? -eq 0 ]; then
        log_success "Docker image pushed successfully"
    else
        log_error "Docker image push failed"
        exit 1
    fi
}

test_image_locally() {
    log_info "Testing Docker image locally..."
    
    # Test the image locally
    docker run --rm -d --name test-smolvlm -p 8000:8000 ${FULL_IMAGE_NAME}
    
    # Wait for container to start
    sleep 10
    
    # Test health endpoint
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Local test passed"
        docker stop test-smolvlm
    else
        log_warning "Local test failed or container not ready"
        docker stop test-smolvlm || true
    fi
}

create_runpod_template() {
    log_info "Creating RunPod template configuration..."
    
    cat > runpod-template.json << EOF
{
    "name": "SmolVLM Geotechnical Engineering",
    "description": "SmolVLM vision-language model optimized for geotechnical engineering document analysis",
    "dockerArgs": "",
    "containerDiskInGb": 20,
    "volumeInGb": 10,
    "volumeMountPath": "/app/cache",
    "ports": "8000/http",
    "env": [
        {
            "key": "TRANSFORMERS_CACHE",
            "value": "/app/cache"
        },
        {
            "key": "HF_HOME", 
            "value": "/app/cache"
        },
        {
            "key": "TORCH_HOME",
            "value": "/app/cache"
        },
        {
            "key": "PYTHONUNBUFFERED",
            "value": "1"
        }
    ],
    "image": "${FULL_IMAGE_NAME}",
    "readme": "# SmolVLM Geotechnical Engineering\\n\\nThis template runs SmolVLM-Instruct optimized for geotechnical engineering document analysis.\\n\\n## Input Format\\n\\n\`\`\`json\\n{\\n  \\"input\\": {\\n    \\"image_data\\": \\"base64_encoded_image\\",\\n    \\"query\\": \\"Your analysis query\\",\\n    \\"max_new_tokens\\": 512,\\n    \\"temperature\\": 0.3\\n  }\\n}\\n\`\`\`\\n\\n## Features\\n- Optimized for engineering drawings and technical documents\\n- Support for multiple image formats\\n- Configurable generation parameters\\n- GPU acceleration with flash attention\\n\\nPowered by SmolVLM-Instruct and RunPod Serverless GPU."
}
EOF
    
    log_success "RunPod template configuration created: runpod-template.json"
}

create_deployment_guide() {
    log_info "Creating deployment guide..."
    
    cat > RUNPOD_DEPLOYMENT.md << 'EOF'
# RunPod Deployment Guide

## Prerequisites

1. RunPod account: https://runpod.io
2. Docker Hub or other container registry account
3. Docker installed locally

## Step 1: Build and Push Container

```bash
# Update the registry name in deploy-runpod.sh
# Replace "your-registry" with your actual registry

# Run deployment script
chmod +x deploy-runpod.sh
./deploy-runpod.sh build push
```

## Step 2: Create RunPod Template

1. Go to https://runpod.io/console/serverless/user/templates
2. Click "New Template"
3. Use the configuration from `runpod-template.json`
4. Or upload the JSON file directly

### Template Settings:
- **Name**: SmolVLM Geotechnical Engineering
- **Image**: your-registry/geotechnical-smolvlm-runpod:v1.0.0
- **Container Disk**: 20 GB
- **Volume**: 10 GB
- **Ports**: 8000/http

## Step 3: Deploy Serverless Endpoint

1. Go to https://runpod.io/console/serverless
2. Click "New Endpoint"
3. Select your template
4. Configure:
   - **Name**: geotechnical-smolvlm
   - **Min Workers**: 0 (for cost optimization)
   - **Max Workers**: 10 (adjust based on needs)
   - **Idle Timeout**: 5 seconds
   - **Flash Boot**: Enable (for faster cold starts)
   - **GPU**: RTX 4090 or A100 (recommended)

## Step 4: Get API Credentials

1. Go to https://runpod.io/console/user/settings
2. Create API Key
3. Note your Endpoint ID from the serverless console

## Step 5: Configure Streamlit App

Add to your `.streamlit/secrets.toml`:

```toml
RUNPOD_API_KEY = "your_api_key_here"
RUNPOD_ENDPOINT_ID = "your_endpoint_id_here"
```

Or set environment variables:

```bash
export RUNPOD_API_KEY="your_api_key_here"
export RUNPOD_ENDPOINT_ID="your_endpoint_id_here"
```

## Testing

Test your endpoint:

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": {
      "image_data": "base64_encoded_image_here",
      "query": "Analyze this engineering document",
      "max_new_tokens": 512,
      "temperature": 0.3
    }
  }'
```

## Cost Optimization Tips

1. **Use Min Workers = 0**: Only pay when processing
2. **Enable Flash Boot**: Faster startup, lower costs
3. **Set appropriate Idle Timeout**: Balance responsiveness vs cost
4. **Choose right GPU**: RTX 4090 for most workloads, A100 for heavy batch processing
5. **Monitor usage**: Use RunPod dashboard to track costs

## Estimated Costs

- **RTX 4090**: ~$0.00011-0.00016 per second
- **A100 40GB**: ~$0.0004-0.0006 per second
- **Cold start**: ~2-3 seconds with Flash Boot
- **Typical inference**: 3-8 seconds depending on query complexity

## Troubleshooting

### Common Issues:

1. **Container won't start**:
   - Check container logs in RunPod console
   - Verify image is accessible
   - Check disk space requirements

2. **Model loading fails**:
   - Ensure sufficient GPU memory
   - Check HuggingFace token if using private models
   - Verify transformers version compatibility

3. **Slow performance**:
   - Use appropriate GPU type
   - Enable Flash Boot
   - Consider pre-downloading model in container

4. **API errors**:
   - Verify API key and endpoint ID
   - Check request format
   - Monitor rate limits

## Support

- RunPod Documentation: https://docs.runpod.io
- RunPod Discord: https://discord.gg/runpod
- HuggingFace SmolVLM: https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct
EOF
    
    log_success "Deployment guide created: RUNPOD_DEPLOYMENT.md"
}

show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build           Build Docker image"
    echo "  push            Push Docker image to registry" 
    echo "  test            Test Docker image locally"
    echo "  template        Create RunPod template configuration"
    echo "  guide          Create deployment guide"
    echo "  all            Run all steps (build, push, test, template, guide)"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 build push"
    echo "  $0 all"
}

# Main execution
main() {
    case "$1" in
        "build")
            check_dependencies
            build_image
            ;;
        "push")
            push_image
            ;;
        "test")
            test_image_locally
            ;;
        "template")
            create_runpod_template
            ;;
        "guide")
            create_deployment_guide
            ;;
        "all")
            check_dependencies
            build_image
            test_image_locally
            push_image
            create_runpod_template
            create_deployment_guide
            log_success "All deployment steps completed!"
            log_info "Next steps:"
            log_info "1. Update your registry name in the script"
            log_info "2. Push your image: ./deploy-runpod.sh push"
            log_info "3. Follow RUNPOD_DEPLOYMENT.md for RunPod setup"
            ;;
        "help"|"--help"|"-h"|"")
            show_usage
            ;;
        *)
            # Handle multiple commands
            for cmd in "$@"; do
                case "$cmd" in
                    "build") build_image ;;
                    "push") push_image ;;
                    "test") test_image_locally ;;
                    "template") create_runpod_template ;;
                    "guide") create_deployment_guide ;;
                    *) log_warning "Unknown command: $cmd" ;;
                esac
            done
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
