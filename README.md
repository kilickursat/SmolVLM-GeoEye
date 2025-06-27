# 🏗️ Tunnelling & Geotechnical Engineering Workflow

A comprehensive multi-modal document processing system using **SmolVLM**, **SmolAgent**, and **Streamlit** for geotechnical engineering workflows.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![AI](https://img.shields.io/badge/AI-SmolVLM%20%7C%20SmolAgent-purple)

## 🌟 Features

### 🤖 **AI-Powered Analysis**
- **Vision-Language Model**: SmolVLM for intelligent document and image analysis
- **Multi-Agent System**: SmolAgent orchestration with specialized engineering agents
- **Natural Language Interface**: Chat-based interaction for technical queries

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
- 4GB RAM minimum (8GB recommended)
- Internet connection for initial model downloads
- Optional: CUDA-compatible GPU for faster processing

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

# Add your HuggingFace token (optional but recommended)
# Edit ~/.streamlit/secrets.toml and add:
# HF_TOKEN = "your_hf_token_here"
```

### 5. Run the Application
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
   HF_TOKEN = "your_hf_token_here"
   ```
4. Deploy with one click!

### 🚀 **Heroku Deployment**
```bash
# Install Heroku CLI, then:
heroku create your-app-name
heroku config:set HF_TOKEN=your_hf_token_here
git push heroku main
```

### 🐳 **Docker Deployment**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### ☁️ **Google Cloud Run**
```bash
gcloud run deploy geotechnical-workflow \
  --source . \
  --platform managed \
  --region us-central1 \
  --set-env-vars HF_TOKEN=your_token
```

## 📖 Usage Guide

### 🔐 **HuggingFace Token Setup**
For optimal performance and faster model downloads:

1. Create account at [HuggingFace](https://huggingface.co/)
2. Generate token at [Settings > Tokens](https://huggingface.co/settings/tokens)
3. Add token to your deployment:
   - **Local**: Add to `~/.streamlit/secrets.toml`
   - **Streamlit Cloud**: Add to secrets in dashboard
   - **Heroku**: `heroku config:set HF_TOKEN=your_token`
   - **Environment Variable**: `export HF_TOKEN=your_token`

### 📁 **Document Upload**
1. Use the sidebar file uploader
2. Supported formats: PDF, PNG, JPG, JPEG, CSV, XLSX, JSON, MD
3. Click "Process Document" to analyze
4. Documents are stored in session (cleared on refresh)

### 💬 **AI Chat Interface**
Ask questions using natural language:
- **"Analyze the soil data from the uploaded CSV"**
- **"Calculate tunnel support for 6m diameter at 30m depth in fair rock"**
- **"Generate safety checklist for excavation project"**
- **"What are the key findings from the uploaded document?"**

### 📊 **Data Analysis**
1. Go to "Data Analysis" tab
2. Select processed document
3. View statistical summaries and insights
4. Explore document content and metadata

### 📈 **Visualizations**
1. Navigate to "Visualizations" tab
2. Select CSV/Excel document
3. Choose chart type (auto, histogram, scatter, correlation)
4. Generate interactive Plotly charts

## 🤖 AI Agent System

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
The system automatically routes queries to appropriate agents based on content analysis:
- Soil-related queries → Data Processing Agent
- Tunnel/structural queries → Engineering Analysis Agent
- General queries → Most appropriate agent

## 🛠️ Configuration

### **Environment Variables**
```bash
# Required
HF_TOKEN=your_huggingface_token

# Optional
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
TORCH_DEVICE=cuda  # or cpu
LOG_LEVEL=INFO
```

### **Streamlit Configuration**
Located in `~/.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

### **Model Configuration**
The system uses these models by default:
- **Vision-Language**: `HuggingFaceTB/SmolVLM-Instruct`
- **Agent Backend**: `InferenceClientModel` (Hugging Face API)

## 🔧 Advanced Setup

### **GPU Acceleration**
For CUDA-enabled systems:
```bash
# Install CUDA-specific PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Optional: Install FlashAttention for faster inference
pip install flash-attn --no-build-isolation
```

### **Memory Optimization**
For systems with limited RAM:
```python
# In app.py, modify model loading:
torch_dtype = torch.float16  # Use half precision
```

### **Custom Models**
To use different models, modify in `app.py`:
```python
# Change model name in ExtractionModule
model_name = "HuggingFaceTB/SmolVLM-256M"  # Smaller variant
```

## 📋 System Requirements

### **Minimum Requirements**
- Python 3.8+
- 4GB RAM
- 5GB storage (for models)
- Internet connection (first run)

### **Recommended Requirements**
- Python 3.9+
- 8GB RAM
- 10GB storage
- CUDA-compatible GPU
- Stable internet connection

### **Supported Operating Systems**
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+)

## 🐛 Troubleshooting

### **Common Issues**

#### Model Loading Errors
```bash
Error: Could not load SmolVLM model
Solution: 
- Check internet connection
- Verify HuggingFace token
- Ensure sufficient disk space (5GB)
```

#### Memory Issues
```bash
Error: CUDA out of memory
Solution:
- Use CPU mode: TORCH_DEVICE=cpu
- Close other applications
- Use smaller model variant
```

#### Installation Issues
```bash
Error: Package installation failed
Solution:
- Update pip: pip install --upgrade pip
- Use Python 3.9+ for better compatibility
- Try installing packages individually
```

#### SmolAgent Errors
```bash
Error: DocstringParsingException
Solution:
- Ensure docstrings follow exact format
- Check tool function signatures
- Verify smolagents version compatibility
```

### **Performance Optimization**
- **Use GPU**: Significant speedup for vision analysis
- **HuggingFace Token**: Faster model downloads
- **Memory Management**: Close unused browser tabs
- **Network**: Stable connection for model downloads

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### **Development Setup**
```bash
git clone https://github.com/your-username/geotechnical-workflow.git
cd geotechnical-workflow
pip install -r requirements.txt
pip install -e .  # Development install
```

### **Running Tests**
```bash
pytest tests/
```

### **Code Style**
```bash
black app.py  # Format code
flake8 app.py  # Check style
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HuggingFace** for SmolVLM and Transformers
- **Streamlit** for the amazing web framework
- **SmolAgent** for intelligent agent orchestration
- **Plotly** for interactive visualizations
- **Engineering Community** for domain expertise

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/your-username/geotechnical-workflow/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-username/geotechnical-workflow/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/geotechnical-workflow/discussions)
- **Email**: support@your-domain.com

## 🗺️ Roadmap

### **Version 2.1 (Planned)**
- [ ] 3D visualization capabilities
- [ ] Additional AI models (Claude, GPT-4V)
- [ ] Database integration
- [ ] REST API endpoints

### **Version 2.2 (Future)**
- [ ] Multi-language support
- [ ] Advanced report generation
- [ ] Real-time collaboration
- [ ] Mobile application

---

## 🚀 **Ready to Transform Your Geotechnical Engineering Workflow?**

1. **Star this repository** ⭐
2. **Deploy in 5 minutes** with Streamlit Cloud
3. **Upload your first document** and experience the power of AI-driven engineering analysis!

**Built with ❤️ for the Engineering Community**

---

*For the latest updates and releases, visit our [GitHub repository](https://github.com/your-username/geotechnical-workflow).*
