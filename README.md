# 🏗️ SmolVLM-GeoEye: Production-Ready Geotechnical Engineering AI

> **Version 3.2.0** - Enterprise-grade AI workflow for geotechnical engineering powered by SmolVLM Vision-Language Model

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)

## 🚀 Overview

SmolVLM-GeoEye is a production-ready, AI-powered geotechnical engineering application that leverages the SmolVLM Vision-Language Model for advanced document analysis, data extraction, and engineering insights. The system provides a complete workflow from document ingestion to actionable engineering recommendations.

### ✨ Latest Updates (v3.2.0)

🔥 **Major Data Extraction Improvements**:
- **Enhanced SmolVLM Integration**: Now requests structured JSON output for reliable data extraction
- **Robust Parsing**: Multiple fallback mechanisms for data extraction (JSON → regex → text analysis)
- **Improved Agent Intelligence**: AI agents now receive properly structured data for analysis
- **Better Error Handling**: Comprehensive validation and user feedback
- **Real-time Feedback**: Shows extraction results and data counts in UI

### 🎯 Key Features

- **🤖 SmolVLM Vision AI**: Advanced image analysis for geotechnical documents with structured JSON output
- **📊 Intelligent Data Extraction**: Automatic extraction of SPT values, bearing capacity, soil properties
- **🧠 AI Engineering Agents**: Specialized agents for soil analysis, tunnel engineering, and safety assessment
- **📈 Advanced Visualizations**: Interactive charts and 3D soil profiles
- **💰 Cost Tracking**: Real-time monitoring of GPU usage and costs
- **🔄 Auto-Scaling**: Dynamic worker management based on demand
- **🗄️ Persistent Storage**: PostgreSQL database with full data history
- **⚡ High Performance**: Redis caching and async processing
- **🔒 Production Security**: API authentication and rate limiting
- **📊 Monitoring**: Prometheus metrics and Grafana dashboards

## 🏗️ Architecture

```
┌─────────────────────┐
│  Streamlit Frontend │
├─────────────────────┤
│    FastAPI REST     │
├─────────────────────┤
│   Core Modules      │
│ ┌─────────────────┐ │
│ │ SmolVLM Client  │ │
│ │ Data Extraction │ │
│ │ AI Agents       │ │
│ │ Visualization   │ │
│ └─────────────────┘ │
├─────────────────────┤
│  Infrastructure     │
│ ┌─────┬─────┬─────┐ │
│ │ DB  │Redis│Monitor│
│ └─────┴─────┴─────┘ │
└─────────────────────┘
```

## 📋 Prerequisites

- Docker & Docker Compose
- RunPod account with API access
- GPU-enabled RunPod endpoint with SmolVLM deployed
- 8GB+ RAM for local development
- Python 3.8+ (for development)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/kilickursat/SmolVLM-GeoEye.git
cd SmolVLM-GeoEye
```

### 2. Configure Environment
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
# RunPod Configuration (Required)
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id

# Security (Required for production)
SECRET_KEY=your_secret_key_here
DB_PASSWORD=strong_database_password

# Optional Configuration
ENVIRONMENT=production
COST_ALERT_THRESHOLD=100.0
MAX_WORKERS=10
MIN_WORKERS=0
```

### 3. Test Installation
```bash
# Run comprehensive test suite
python test_smolvlm_extraction.py
```

### 4. Deploy with Docker
```bash
# Make deployment script executable
chmod +x deploy-production.sh

# Run deployment
./deploy-production.sh production
```

### 5. Access Application
- **Main App**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Monitoring**: http://localhost:3000 (Grafana)

## 🔧 Development Setup

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run comprehensive test suite
python test_smolvlm_extraction.py

# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=modules --cov-report=html
```

## 📚 API Documentation

### Authentication
All API endpoints require Bearer token authentication:
```bash
curl -H "Authorization: Bearer YOUR_SECRET_KEY" \
  http://localhost:8000/api/v1/health
```

### Key Endpoints

#### Upload Document
```bash
POST /api/v1/documents/upload
Content-Type: multipart/form-data

curl -X POST \
  -H "Authorization: Bearer YOUR_SECRET_KEY" \
  -F "file=@soil_report.pdf" \
  http://localhost:8000/api/v1/documents/upload
```

#### Analyze Document
```bash
POST /api/v1/documents/{document_id}/analyze
Content-Type: application/json

{
  "query": "What are the SPT values and bearing capacity?",
  "agent_type": "soil"
}
```

#### Query Multiple Documents
```bash
POST /api/v1/query
Content-Type: application/json

{
  "query": "Compare soil properties across all documents",
  "document_ids": [1, 2, 3],
  "include_visualizations": true
}
```

## 🏗️ Module Architecture

### Core Modules

#### 📊 Data Extraction (`modules/data_extraction.py`)
- Advanced pattern matching for geotechnical parameters
- Supports SPT, bearing capacity, density, moisture content, etc.
- Statistical analysis of extracted values
- Confidence scoring for each extraction

#### 🤖 AI Agents (`modules/agents.py`)
- **SoilAnalysisAgent**: Soil mechanics and foundation analysis
- **TunnelSupportAgent**: Tunnel engineering and rock mechanics
- **SafetyChecklistAgent**: Risk assessment and safety protocols
- **GeotechnicalAgentOrchestrator**: Intelligent query routing

#### 📈 Visualization Engine (`modules/visualization.py`)
- SPT depth profiles with soil consistency zones
- Parameter distribution charts
- Correlation matrices
- 3D soil profile visualization
- Time series analysis

#### 💰 Cost Tracking (`modules/smolvlm_client.py`)
- Real-time cost monitoring
- Per-job cost estimation
- Worker utilization metrics
- Auto-scaling recommendations

## 🔍 Extracted Parameters

The system automatically extracts:
- **SPT Values**: N-values with depth correlation
- **Bearing Capacity**: Ultimate and allowable values
- **Soil Properties**: Density, moisture content, void ratio
- **Strength Parameters**: Cohesion, friction angle, UCS
- **Atterberg Limits**: Liquid limit, plastic limit, PI
- **Settlement Data**: Immediate and consolidation
- **Rock Quality**: RQD values, rock strength

## 📊 Monitoring & Analytics

### Prometheus Metrics
- Request count and duration
- Worker status and utilization
- Cache hit/miss ratios
- Error rates by type
- Cost accumulation

### Grafana Dashboards
- System performance overview
- Cost analysis and projections
- User activity tracking
- Error analysis and debugging

## 🔒 Security Features

- API token authentication
- Rate limiting (100 requests/minute)
- CORS configuration
- Input validation and sanitization
- Secure file handling
- Database encryption at rest

## 🚀 Production Deployment

### Cloud Deployment (AWS/GCP/Azure)
1. Set up managed PostgreSQL and Redis
2. Configure load balancer with SSL
3. Deploy using Kubernetes or ECS
4. Set up monitoring and alerting

### Scaling Considerations
- Horizontal scaling for API servers
- GPU worker auto-scaling on RunPod
- Database read replicas for high load
- CDN for static assets

## 🧪 Testing Strategy

### Unit Tests
- Core module functionality
- Data extraction accuracy
- Agent decision logic
- Cache operations

### Integration Tests
- End-to-end document processing
- API endpoint validation
- Database operations
- RunPod communication

### Performance Tests
- Load testing with multiple users
- Large document processing
- Concurrent API requests
- Memory and CPU profiling

## 🔧 Troubleshooting

### Common Issues

#### SmolVLM Data Extraction Not Working
```bash
# Run extraction test
python test_smolvlm_extraction.py

# Check extraction results
python -c "
from modules.data_extraction import EnhancedGeotechnicalDataExtractor
extractor = EnhancedGeotechnicalDataExtractor()
result = extractor.extract_numerical_data_from_text('SPT N-value: 15 at 3m depth')
print(f'Extracted: {len(result)} parameters')
"
```

#### RunPod Connection Failed
```bash
# Check endpoint status
python enhanced_monitor_runpod.py --once

# Verify API credentials
curl -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/health
```

#### Database Connection Error
```bash
# Check PostgreSQL status
docker-compose ps db
docker-compose logs db

# Test connection
docker-compose exec db psql -U postgres -d geotechnical
```

#### High Cost Alert
```bash
# Review cost analytics
python enhanced_monitor_runpod.py --report --hours 24

# Adjust worker scaling
docker-compose exec app python -c "
from modules.smolvlm_client import EnhancedRunPodClient
client.scale_workers(2)  # Scale to 2 workers
"
```

## 📈 Performance Optimization

### Caching Strategy
- Document analysis results cached for 2 hours
- Numerical data cached for 24 hours
- System health cached for 30 seconds

### Database Optimization
- Indexed queries on document_id and timestamp
- Partitioned tables for metrics data
- Regular vacuum and analyze

### Worker Management
- Auto-scale based on queue depth
- Minimum 0 workers during idle
- Maximum 10 workers during peak

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Run linting and tests before PR

## 📜 License

This project is licensed under the GNU General Public License v3.0 - see [LICENSE](LICENSE) for details.

## 🔗 Resources

- **SmolVLM Model**: [HuggingFace](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- **RunPod Platform**: [RunPod.io](https://runpod.io)
- **Documentation**: [Wiki](https://github.com/kilickursat/SmolVLM-GeoEye/wiki)
- **Issues**: [GitHub Issues](https://github.com/kilickursat/SmolVLM-GeoEye/issues)

## 📊 Changelog

### v3.2.0 - Enhanced Data Extraction (Current)
- 🔥 **Major SmolVLM Integration Fixes**:
  - Enhanced query to request structured JSON output
  - Robust JSON parsing with multiple fallback mechanisms
  - Improved agent integration with properly structured data
  - Real-time extraction feedback and data counts
  - Comprehensive error handling and user warnings
- ✅ Comprehensive test suite (`test_smolvlm_extraction.py`)
- ✅ Better UI feedback for extraction results
- ✅ Enhanced chat interface with data validation
- ✅ Fixed session state initialization issues

### v3.1.0 - Production Release
- ✅ Complete module architecture implementation
- ✅ Enhanced data extraction with confidence scoring
- ✅ AI agents for specialized analysis
- ✅ Real-time cost tracking and monitoring
- ✅ PostgreSQL and Redis integration
- ✅ FastAPI REST endpoints
- ✅ Comprehensive unit tests
- ✅ Production deployment scripts
- ✅ Docker Compose orchestration
- ✅ Prometheus/Grafana monitoring

### v3.0.0 - Enhanced Features
- 🚀 SmolVLM integration
- 📊 Basic visualization
- 💬 Chat interface

---

**Built with ❤️ for the geotechnical engineering community**

*Powered by SmolVLM - Bringing AI vision to engineering*
