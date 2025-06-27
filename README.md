# 🏗️ Tunnelling & Geotechnical Engineering Workflow

A comprehensive multi-modal document processing system using **SmolVLM**, **SmolAgent**, **RunPod Serverless GPU**, and **Streamlit** for geotechnical engineering workflows.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![RunPod](https://img.shields.io/badge/RunPod-Serverless%20GPU-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![AI](https://img.shields.io/badge/AI-SmolVLM%20%7C%20SmolAgent-orange)

## 🌟 Features

### 🚀 **RunPod Serverless GPU Integration**
- **Serverless GPU Computing**: Auto-scaling SmolVLM on RTX 4090, A100, H100 GPUs
- **Pay-per-Request**: Cost-effective inference with millisecond billing
- **FlashBoot Technology**: 2-3 second cold starts for instant availability
- **Global Edge Network**: Low-latency processing from multiple regions

### 🤖 **AI-Powered Analysis**
- **Vision-Language Model**: SmolVLM-Instruct (2B) optimized for engineering content
- **Multi-Agent System**: SmolAgent orchestration with specialized engineering agents
- **Natural Language Interface**: Chat-based interaction for technical queries
- **GPU Acceleration**: High-performance inference via RunPod serverless

### 📁 **Multi-Modal Document Processing**
- **PDF Documents**: Technical reports, specifications, drawings
- **Images**: Engineering drawings, site photos, charts (PNG, JPG, JPEG)
- **Structured Data**: CSV, Excel files with test results and measurements
- **Text Files**: JSON, Markdown, plain text

### 🛠️ **Engineering-Specific Tools**
- **Soil Analysis Agent**: Automated soil test data interpretation
- **Tunnel Engineering Agent**: Support calculations and safety assessments
- **Visualization Engine**: Interactive charts and statistical analysis
- **Safety Checklist Generator**: Project-specific safety protocols

### 📊 **Advanced Analytics**
- **Statistical Analysis**: Descriptive statistics and correlation analysis
- **Interactive Visualizations**: Plotly-powered charts and graphs
- **Data Processing**: Pandas-based data manipulation and analysis
- **Real-time Chat Interface**: Streamlit-based conversational UI

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- RunPod account ([Sign up](https://runpod.io))
- 4GB RAM minimum (8GB recommended)
- Internet connection for RunPod integration

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/geotechnical-workflow.git
cd geotechnical-workflow
```

### 2. Create Virtual Environment
```bash
python -m venv geotechnical_env

# Windows
geotechnical_env\Scripts\activate

# macOS/Linux
source geotechnical_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Configuration
```bash
# Run setup script
chmod +x setup.sh
./setup.sh
```

### 5. Configure RunPod Integration

#### 5.1 Deploy SmolVLM to RunPod
```bash
# Build and deploy container
docker build -f Dockerfile -t your-registry/smolvlm-geotechnical:latest .
docker push your-registry/smolvlm-geotechnical:latest

# Or use the deployment script
chmod +x runpod-deploy.sh
./runpod-deploy.sh all
```

#### 5.2 Create RunPod Serverless Endpoint
1. Go to [RunPod Serverless Console](https://runpod.io/console/serverless)
2. Create new template with your Docker image
3. Deploy endpoint with GPU (RTX 4090+ recommended)
4. Note your **Endpoint ID** and **API Key**

#### 5.3 Configure Credentials
Add to `.streamlit/secrets.toml`:
```toml
RUNPOD_API_KEY = "your_runpod_api_key_here"
RUNPOD_ENDPOINT_ID = "your_runpod_endpoint_id_here"
HF_TOKEN = "your_huggingface_token_here"
```

### 6. Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## 🌐 Deployment Options

### 🔥 **Streamlit Cloud** (Recommended)
1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add secrets in Streamlit Cloud dashboard:
   ```toml
   RUNPOD_API_KEY = "your_runpod_api_key_here"
   RUNPOD_ENDPOINT_ID = "your_runpod_endpoint_id_here"
   HF_TOKEN = "your_hf_token_here"
   ```
4. Deploy with one click!

### 🚀 **Heroku Deployment**
```bash
# Install Heroku CLI, then:
heroku create your-app-name
heroku config:set RUNPOD_API_KEY=your_runpod_api_key_here
heroku config:set RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id_here
heroku config:set HF_TOKEN=your_hf_token_here
git push heroku main
```

### 🐳 **Docker Deployment**
```bash
# Build and run locally
docker-compose up --build

# Or with RunPod integration
docker-compose -f docker-compose.runpod.yml up --build
```

### ☁️ **Google Cloud Run**
```bash
gcloud run deploy geotechnical-workflow \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars RUNPOD_API_KEY=your_api_key,RUNPOD_ENDPOINT_ID=your_endpoint_id
```

## 📖 Usage Guide

### 🚀 **RunPod Serverless GPU Setup**

#### Complete Setup Guide
1. **Create RunPod Account**: [Sign up here](https://runpod.io)
2. **Build Container**: Use provided Dockerfile and deployment scripts
3. **Deploy Endpoint**: Create serverless endpoint with GPU acceleration
4. **Configure App**: Add credentials to Streamlit secrets
5. **Test Integration**: Use provided testing utilities

#### Benefits of RunPod Integration
- **🚀 Performance**: 10-100x faster than CPU inference
- **💰 Cost-Effective**: Pay only for actual usage (no idle costs)
- **⚡ Auto-Scaling**: Scale from 0 to thousands of requests instantly
- **🔥 Fast Cold Starts**: FlashBoot technology for 2-3 second startup
- **🌍 Global Availability**: Multiple regions for low latency

### 📁 **Document Upload**
1. Use the sidebar file uploader
2. Supported formats: PDF, PNG, JPG, JPEG, CSV, XLSX, JSON, MD
3. Click "Process Document" to analyze
4. Documents are enhanced with RunPod AI analysis when available
5. View processing status and AI analysis results

### 💬 **AI Chat Interface**
Ask questions using natural language:
- **"Analyze the soil data from the uploaded CSV"**
- **"Calculate tunnel support for 6m diameter at 30m depth in fair rock"**
- **"Generate safety checklist for excavation project"**
- **"What are the key findings from the uploaded engineering drawing?"**
- **"Analyze the structural details in this image"**

### 📊 **Data Analysis**
1. Go to "Data Analysis" tab
2. Select processed document
3. View AI analysis results from RunPod SmolVLM
4. Explore statistical summaries and insights
5. Access document content and metadata

### 📈 **Visualizations**
1. Navigate to "Visualizations" tab
2. Select CSV/Excel document
3. Choose chart type (auto, histogram, scatter, correlation)
4. Generate interactive Plotly charts

## 🤖 AI Agent System

### **RunPod Vision Agent**
- **Function**: Advanced vision-language analysis using SmolVLM on GPU
- **Capabilities**: 
  - Engineering drawing analysis
  - Technical specification extraction
  - Safety assessment from images
  - Material property identification
  - Structural detail recognition

### **Data Processing Agent**
- **Function**: Analyzes soil test data and material properties
- **Tools**: `analyze_soil_data()`
- **Capabilities**: 
  - Soil density analysis
  - Moisture content evaluation
  - Bearing capacity assessment
  - Engineering recommendations

### **Engineering Analysis Agent**
- **Function**: Structural and tunneling calculations
- **Tools**: `calculate_tunnel_support()`, `generate_safety_checklist()`
- **Capabilities**:
  - Tunnel support design
  - Safety protocol generation
  - Risk assessment
  - Code compliance checking

### **Query Routing Intelligence**
The system automatically routes queries to appropriate agents:
- Vision-related queries → RunPod SmolVLM Agent
- Soil-related queries → Data Processing Agent
- Tunnel/structural queries → Engineering Analysis Agent
- General queries → Most appropriate agent

## 🧪 Testing and Monitoring

### **Test RunPod Integration**
```bash
# Test connection
python test_runpod.py --test-connection

# Test vision analysis
python test_runpod.py --test-vision

# Run performance benchmark
python test_runpod.py --benchmark

# Monitor endpoint health
python test_runpod.py --monitor 10

# Run all tests
python test_runpod.py --all
```

### **Automated Testing**
```bash
# Run unit tests
pytest tests/

# Check code quality
black app.py
flake8 app.py

# Run integration tests
python -m pytest tests/ -m integration
```

## 🔧 Configuration

### **Environment Variables**
```bash
# Required for RunPod Integration
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id

# Optional
HF_TOKEN=your_huggingface_token
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
```

### **RunPod Configuration**
```json
{
  "name": "SmolVLM Geotechnical Engineering",
  "image": "your-registry/smolvlm-geotechnical:latest",
  "containerDiskInGb": 10,
  "volumeInGb": 20,
  "ports": "8000/http",
  "env": [
    {"key": "MODEL_NAME", "value": "HuggingFaceTB/SmolVLM-Instruct"},
    {"key": "MAX_NEW_TOKENS", "value": "512"},
    {"key": "TEMPERATURE", "value": "0.3"}
  ]
}
```

### **Streamlit Configuration**
Located in `~/.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

## 📋 System Requirements

### **Minimum Requirements**
- Python 3.8+
- 4GB RAM
- 2GB storage
- Internet connection (for RunPod)
- RunPod account

### **Recommended Requirements**
- Python 3.9+
- 8GB RAM
- 5GB storage
- Stable internet connection
- RunPod account with GPU access

### **RunPod GPU Options**
- **RTX 4090**: Best price/performance for most workloads
- **A100 40GB**: High memory for large batch processing
- **H100**: Maximum performance for demanding applications

## 🐛 Troubleshooting

### **Common Issues**

#### RunPod Integration Issues
```bash
Error: RunPod status shows "Not Configured"
Solution: 
- Verify API key and endpoint ID in secrets
- Check RunPod endpoint status in console
- Test connection with: python test_runpod.py --test-connection
```

#### Vision Analysis Failures
```bash
Error: Vision analysis fails or times out
Solution:
- Check RunPod endpoint logs
- Verify GPU availability
- Increase timeout in requests
- Monitor endpoint with: python test_runpod.py --monitor 5
```

#### Performance Issues
```bash
Error: Slow response times
Solution:
- Enable FlashBoot for faster cold starts
- Use RTX 4090+ GPUs for better performance
- Set minimum workers to 1 for instant availability
- Optimize Docker image size
```

#### Model Loading Errors
```bash
Error: Could not load SmolVLM model
Solution: 
- Check internet connection
- Verify RunPod container logs
- Ensure sufficient GPU memory
- Check HuggingFace token validity
```

### **Performance Optimization**
- **RunPod GPU Selection**: Use RTX 4090+ for optimal performance
- **Auto-Scaling**: Configure based on usage patterns
- **FlashBoot**: Enable for 2-3 second cold starts
- **Batch Processing**: Group requests when possible
- **Image Optimization**: Compress Docker images for faster deployment

## 💰 Cost Optimization

### **RunPod Pricing Tips**
- **Pay-per-Request**: Only pay for actual inference time
- **Auto-Scaling**: Scale to zero when not in use
- **GPU Selection**: Choose appropriate GPU for workload
- **Batch Processing**: Process multiple requests together
- **Regional Selection**: Use closest region to reduce latency costs

### **Estimated Costs** (as of 2024)
- **RTX 4090**: ~$0.00011-0.00016 per second
- **A100 40GB**: ~$0.0004-0.0006 per second
- **H100**: ~$0.0008-0.0012 per second

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### **Development Setup**
```bash
git clone https://github.com/your-username/geotechnical-workflow.git
cd geotechnical-workflow
pip install -r requirements.txt
pip install -e .  # Development install

# Set up pre-commit hooks
pre-commit install
```

### **Development Workflow**
```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes and test
python test_runpod.py --all
pytest tests/

# Format and lint
black .
flake8 .

# Commit and push
git commit -m "Add your feature"
git push origin feature/your-feature
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **RunPod** for providing serverless GPU infrastructure
- **HuggingFace** for SmolVLM and Transformers
- **Streamlit** for the amazing web framework
- **SmolAgent** for intelligent agent orchestration
- **Plotly** for interactive visualizations
- **Engineering Community** for domain expertise

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/your-username/geotechnical-workflow/wiki)
- **RunPod Setup**: [docs/RUNPOD_SETUP.md](docs/RUNPOD_SETUP.md)
- **Issues**: [GitHub Issues](https://github.com/your-username/geotechnical-workflow/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/geotechnical-workflow/discussions)
- **RunPod Support**: [RunPod Discord](https://discord.gg/runpod)

## 🗺️ Roadmap

### **Version 2.2 (Current - RunPod Integration)**
- [x] RunPod serverless GPU integration
- [x] Enhanced SmolVLM vision analysis
- [x] Auto-scaling infrastructure
- [x] Performance optimization
- [x] Comprehensive testing utilities

### **Version 2.3 (Planned)**
- [ ] Multi-model support (GPT-4V, Claude Vision)
- [ ] Enhanced document processing pipeline
- [ ] Real-time collaboration features
- [ ] Advanced caching and optimization

### **Version 2.4 (Future)**
- [ ] 3D visualization capabilities
- [ ] Database integration for persistent storage
- [ ] REST API endpoints
- [ ] Mobile application
- [ ] Multi-language support

---

## 🚀 **Ready to Transform Your Geotechnical Engineering Workflow?**

1. **⭐ Star this repository**
2. **🚀 Deploy with RunPod Serverless GPU** for 10-100x performance boost
3. **📁 Upload your first engineering document** and experience AI-powered analysis
4. **💬 Chat with your documents** using natural language
5. **📊 Generate insights** with automated analysis and visualizations

### **Next Steps:**
```bash
# Quick start with RunPod
git clone https://github.com/your-username/geotechnical-workflow.git
cd geotechnical-workflow
./setup.sh
# Configure RunPod credentials
streamlit run app.py
```

**Built with ❤️ for the Engineering Community**

**Enhanced with 🚀 RunPod Serverless GPU Technology**

---

*For the latest updates and releases, visit our [GitHub repository](https://github.com/your-username/geotechnical-workflow).*

*Experience the power of serverless GPU acceleration with RunPod - deploy in minutes, scale instantly, pay only for what you use.*
