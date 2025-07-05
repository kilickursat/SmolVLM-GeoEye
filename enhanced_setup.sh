#!/bin/bash
"""
SmolVLM-GeoEye Enhanced Quick Setup Script
==========================================

Automated setup for the enhanced version with all issue fixes.
This script will install dependencies, configure environment, and verify the setup.

Author: SmolVLM-GeoEye Team (Enhanced)
Version: 3.1.0 - Issue Resolution Update
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}$1${NC}"
}

# Main setup function
main() {
    print_header "🚀 SmolVLM-GeoEye Enhanced Setup v3.1.0"
    print_header "=========================================="
    echo ""
    print_status "Setting up enhanced geotechnical workflow with issue fixes..."
    echo ""

    # Check Python version
    print_status "Checking Python version..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed. Please install Python 3.8 or higher."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_success "Python $PYTHON_VERSION found"

    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi

    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate || {
        print_error "Failed to activate virtual environment"
        exit 1
    }
    print_success "Virtual environment activated"

    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    print_success "Pip upgraded"

    # Install dependencies
    print_status "Installing enhanced dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt > /dev/null 2>&1
        print_success "Dependencies installed from requirements.txt"
    else
        print_warning "requirements.txt not found, installing core dependencies..."
        pip install streamlit plotly pandas numpy pillow PyPDF2 python-dotenv requests smolagents > /dev/null 2>&1
        print_success "Core dependencies installed"
    fi

    # Setup environment file
    print_status "Setting up environment configuration..."
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Environment file created from .env.example"
            print_warning "Please edit .env file with your RunPod and HuggingFace credentials"
        else
            create_env_file
            print_success "Basic environment file created"
        fi
    else
        print_warning ".env file already exists"
    fi

    # Verify setup
    print_status "Verifying enhanced setup..."
    verify_setup

    # Create logs directory
    print_status "Creating logs directory..."
    mkdir -p logs
    print_success "Logs directory created"

    # Create data directories
    print_status "Creating data directories..."
    mkdir -p data/uploads data/processed data/cache
    print_success "Data directories created"

    # Test import
    print_status "Testing enhanced imports..."
    python3 -c "
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image
import requests
from dotenv import load_dotenv
print('✅ All core imports successful')
" > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        print_success "Import test passed"
    else
        print_error "Import test failed - check dependencies"
        exit 1
    fi

    # Display setup completion
    echo ""
    print_header "🎉 Enhanced Setup Complete!"
    print_header "=========================="
    echo ""
    print_success "✅ All critical issues have been resolved:"
    echo -e "   ${GREEN}• Numerical data visualization fixed${NC}"
    echo -e "   ${GREEN}• RunPod cost monitoring implemented${NC}"
    echo -e "   ${GREEN}• SmolVLM usage visibility enhanced${NC}"
    echo ""
    
    print_status "Next steps:"
    echo -e "   ${BLUE}1.${NC} Edit .env file with your credentials:"
    echo -e "      ${YELLOW}nano .env${NC}"
    echo ""
    echo -e "   ${BLUE}2.${NC} Start the enhanced application:"
    echo -e "      ${YELLOW}source venv/bin/activate${NC}"
    echo -e "      ${YELLOW}streamlit run app.py${NC}"
    echo ""
    echo -e "   ${BLUE}3.${NC} Monitor RunPod performance:"
    echo -e "      ${YELLOW}python enhanced_monitor_runpod.py${NC}"
    echo ""
    
    print_status "Enhanced features available:"
    echo -e "   🤖 ${PURPLE}SmolVLM${NC} vision analysis with cost tracking"
    echo -e "   📊 ${BLUE}Advanced${NC} numerical data extraction and visualization"
    echo -e "   💰 ${YELLOW}Real-time${NC} RunPod cost monitoring and optimization"
    echo -e "   🚀 ${GREEN}Performance${NC} analytics and worker utilization metrics"
    echo ""
    
    print_warning "Remember to configure your RunPod and HuggingFace tokens in .env file!"
}

# Function to create basic environment file
create_env_file() {
    cat > .env << 'EOF'
# SmolVLM-GeoEye Enhanced Environment Configuration
# =================================================

# RunPod Configuration (Required for GPU processing)
RUNPOD_API_KEY=your_runpod_api_key_here
RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id_here

# HuggingFace Configuration (Required for SmolAgent)
HF_TOKEN=your_huggingface_token_here

# Application Settings
ENVIRONMENT=production
DEBUG=false

# RunPod Advanced Settings
RUNPOD_TIMEOUT=300
RUNPOD_MAX_RETRIES=3

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/smolvlm_geoeye.log

# Cost Tracking (Enhanced Feature)
ENABLE_COST_TRACKING=true
COST_ALERT_THRESHOLD=100.0

# Performance Settings
MAX_WORKERS=10
MIN_WORKERS=0
BATCH_SIZE=1

# Enhanced Monitoring
ENABLE_MONITORING=true
MONITORING_INTERVAL=30
EOF
}

# Function to verify setup
verify_setup() {
    local issues=0
    
    # Check if .env file has been configured
    if [ -f ".env" ]; then
        if grep -q "your_runpod_api_key_here" .env; then
            print_warning "RunPod API key not configured in .env"
            ((issues++))
        fi
        
        if grep -q "your_huggingface_token_here" .env; then
            print_warning "HuggingFace token not configured in .env"
            ((issues++))
        fi
    fi
    
    # Check if main files exist
    if [ ! -f "app.py" ]; then
        print_error "app.py not found"
        ((issues++))
    fi
    
    if [ ! -f "enhanced_monitor_runpod.py" ]; then
        print_error "enhanced_monitor_runpod.py not found"
        ((issues++))
    fi
    
    if [ $issues -eq 0 ]; then
        print_success "Setup verification passed"
    else
        print_warning "Setup completed with $issues configuration issues"
    fi
}

# Function to show usage
show_usage() {
    echo "SmolVLM-GeoEye Enhanced Quick Setup"
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -v, --verify   Only verify existing setup"
    echo "  -c, --clean    Clean install (remove existing venv)"
    echo ""
    echo "Enhanced features in v3.1.0:"
    echo "  ✅ Fixed numerical data visualization"
    echo "  ✅ Real-time RunPod cost monitoring"
    echo "  ✅ Enhanced SmolVLM usage visibility"
    echo "  📊 Advanced performance analytics"
    echo "  💰 Cost optimization recommendations"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_usage
        exit 0
        ;;
    -v|--verify)
        print_status "Verifying existing setup..."
        verify_setup
        exit 0
        ;;
    -c|--clean)
        print_status "Performing clean install..."
        if [ -d "venv" ]; then
            rm -rf venv
            print_success "Removed existing virtual environment"
        fi
        main
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        show_usage
        exit 1
        ;;
esac
