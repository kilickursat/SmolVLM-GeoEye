#!/bin/bash

# setup.sh - Streamlit Configuration for Cloud Deployment
# ======================================================

# Create Streamlit configuration directory
mkdir -p ~/.streamlit/

# Create Streamlit configuration file
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

# Create credentials file for Streamlit secrets
cat > ~/.streamlit/secrets.toml << EOF
# Streamlit Secrets Configuration
# Copy this file and add your actual tokens

# HuggingFace Token (Optional but recommended)
# Get your token from: https://huggingface.co/settings/tokens
# HF_TOKEN = "hf_your_token_here"

# Example environment variables
# ENVIRONMENT = "production"
# DEBUG = false
EOF

echo "✅ Streamlit configuration files created!"
echo "📝 Please edit ~/.streamlit/secrets.toml to add your HuggingFace token"
echo "🚀 Ready for deployment!"
