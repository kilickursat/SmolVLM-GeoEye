# 🏗️ SmolVLM-GeoEye: Enhanced Geotechnical Engineering Workflow

> **🚀 MAJOR UPDATE v3.1.0** - All critical issues resolved with enhanced monitoring and visualization

## 🎯 Issue Resolution Summary

### ✅ **RESOLVED ISSUES:**

1. **📊 Numerical Data Visualization Fixed**
   - Enhanced pattern matching for geotechnical parameters
   - Guaranteed visualization of extracted numerical data
   - Real-time statistical analysis and data display
   - Interactive charts with proper scaling and labeling

2. **💰 RunPod Usage Monitoring Enhanced**
   - Real-time cost tracking ($0.00/s issue resolved)
   - Worker status monitoring and utilization metrics
   - Performance analytics with historical trends
   - Automated scaling recommendations

3. **🤖 SmolVLM Usage Visibility Improved**
   - Clear SmolVLM branding throughout UI
   - Usage statistics and success rate tracking
   - Processing time and cost per request display
   - Active status indicators and performance metrics

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your RunPod and HuggingFace tokens
```

### 3. Run Enhanced Application
```bash
streamlit run app.py
```

### 4. Monitor RunPod Performance
```bash
python enhanced_monitor_runpod.py
```

---

## 🔧 New Features & Enhancements

### 📊 **Enhanced Data Analysis**
- **Advanced Pattern Matching**: Extracts SPT values, bearing capacity, density, moisture content, and more
- **Statistical Analysis**: Automatic calculation of min, max, mean, standard deviation
- **Interactive Visualizations**: Dynamic charts that adapt to available data types
- **Data Export**: Export extracted numerical data for further analysis

### 💰 **Cost Tracking & Optimization**
- **Real-time Cost Monitoring**: Track expenses per hour/day/month
- **Worker Utilization Metrics**: Efficiency scoring and optimization suggestions
- **Cost per Job Analysis**: Calculate ROI for each SmolVLM inference
- **Historical Cost Trends**: SQLite database for persistent cost tracking

### 🤖 **SmolVLM Performance Monitoring**
- **Usage Statistics**: Query count, success rate, processing times
- **Status Indicators**: Real-time RunPod and SmolVLM status display
- **Performance Metrics**: Response times, worker availability, job queue status
- **Alert System**: Notifications for performance issues and cost thresholds

### 📈 **Advanced Visualizations**
- **Depth Profile Charts**: SPT N-values vs depth with proper axis scaling
- **Parameter Distribution**: Histograms and box plots for soil properties
- **Correlation Analysis**: Interactive correlation matrices for parameter relationships
- **Multi-document Analysis**: Combined visualizations across all uploaded documents

---

## 🏗️ Geotechnical Engineering Features

### 📄 **Document Processing**
- **PDF Analysis**: Extract text and numerical data from geotechnical reports
- **Image Analysis**: SmolVLM vision AI analyzes engineering drawings and test results
- **CSV/Excel Support**: Process laboratory test data and field measurements
- **Multi-format Support**: JSON, Markdown, and plain text files

### 🔍 **Data Extraction Capabilities**
- **SPT Test Results**: N-values with depth information
- **Soil Properties**: Density, moisture content, plasticity indices
- **Bearing Capacity**: Ultimate and allowable bearing pressures
- **Strength Parameters**: Cohesion, friction angle, UCS values
- **Settlement Data**: Immediate and consolidation settlement values
- **Rock Quality**: RQD values and rock strength parameters

### 🎯 **Engineering Analysis**
- **Soil Classification**: Automatic soil type identification
- **Foundation Recommendations**: Based on extracted soil parameters
- **Settlement Predictions**: Using consolidation and bearing capacity data
- **Safety Factor Calculations**: For foundation design and stability analysis

---

## 📊 Usage Analytics Dashboard

### Real-time Monitoring
```
🚀 RunPod Status: ACTIVE (2 workers ready)
🤖 SmolVLM: 15 queries (93% success rate)
💰 Cost: $0.0234 (12 jobs processed)
📊 Data: 3 docs, 47 numerical values
```

### Performance Metrics
- **Response Times**: Average processing time per document
- **Worker Utilization**: Efficiency of GPU resource usage
- **Cost Optimization**: Recommendations for scaling and cost reduction
- **Success Rates**: SmolVLM inference success statistics

---

## 🔧 Configuration Guide

### Environment Variables
```bash
# RunPod Configuration (Required for GPU processing)
RUNPOD_API_KEY=your_runpod_api_key_here
RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id_here

# HuggingFace Token (Required for SmolAgent)
HF_TOKEN=your_huggingface_token_here

# Optional Settings
RUNPOD_TIMEOUT=300
RUNPOD_MAX_RETRIES=3
LOG_LEVEL=INFO
```

### Cost Management
```bash
# Enable cost tracking and alerts
ENABLE_COST_TRACKING=true
COST_ALERT_THRESHOLD=100.0

# Performance optimization
MAX_WORKERS=10
MIN_WORKERS=0
BATCH_SIZE=1
```

---

## 📈 Monitoring & Analytics

### Enhanced RunPod Monitor
```bash
# Real-time monitoring with cost analysis
python enhanced_monitor_runpod.py

# Generate cost report
python enhanced_monitor_runpod.py --report --hours 24

# Single status check
python enhanced_monitor_runpod.py --once
```

### Features
- **Real-time Cost Tracking**: Monitor expenses as they occur
- **Worker Status**: Live updates on GPU availability and utilization
- **Performance Alerts**: Automatic notifications for issues
- **Historical Analysis**: SQLite database for trend analysis
- **Optimization Recommendations**: AI-powered scaling suggestions

---

## 🎨 User Interface Enhancements

### Status Indicators
- **🚀 RunPod Active**: Green pulsing indicator when GPU workers are ready
- **🤖 SmolVLM Usage**: Purple indicator showing query count and success rate
- **💰 Cost Tracker**: Orange indicator with real-time cost accumulation
- **📊 Data Status**: Blue indicator showing documents and extracted values

### Enhanced Tabs
1. **💬 SmolVLM AI Chat**: Interactive Q&A with document context
2. **📊 Enhanced Data Analysis**: Detailed numerical data breakdown
3. **📈 Advanced Visualizations**: Interactive charts and correlations
4. **🚀 System Performance**: Real-time monitoring dashboard

---

## 🔍 Troubleshooting

### Common Issues

#### Issue: No numerical data visualized
**Solution**: ✅ **FIXED** - Enhanced pattern matching now extracts more data types
```python
# New patterns detect:
- "SPT N=25 at 3.5m depth"
- "bearing capacity = 150 kPa"
- "density = 1.85 g/cm³"
- "moisture content = 12.5%"
```

#### Issue: RunPod showing $0.00/s
**Solution**: ✅ **FIXED** - Real-time cost tracking and worker monitoring
```bash
# Use enhanced monitor
python enhanced_monitor_runpod.py
```

#### Issue: SmolVLM usage not visible
**Solution**: ✅ **FIXED** - Clear indicators throughout UI
- Purple SmolVLM status cards
- Usage statistics in real-time
- Processing time display
- Cost per request tracking

### Performance Optimization
1. **Worker Scaling**: Monitor utilization and scale based on demand
2. **Cost Control**: Set alert thresholds and review daily reports
3. **Data Quality**: Upload high-quality images and well-formatted documents
4. **Batch Processing**: Process multiple documents in sequence for efficiency

---

## 📚 API Reference

### Enhanced RunPod Client
```python
from app import EnhancedRunPodClient, Config

# Initialize with cost tracking
config = Config()
client = EnhancedRunPodClient(config)

# Run with tracking
result = client.run_sync_with_tracking({
    "image_data": base64_image,
    "query": "Analyze geotechnical data",
    "max_new_tokens": 512
})

# Check health with metrics
health = client.enhanced_health_check()
print(f"Workers ready: {health['workers']['ready']}")
print(f"Cost per hour: ${health['cost_per_hour']:.4f}")
```

### Data Extraction
```python
from app import EnhancedGeotechnicalDataExtractor

extractor = EnhancedGeotechnicalDataExtractor()
data = extractor.extract_numerical_data_from_text(text)

# Access extracted parameters
spt_values = data['spt_values']
bearing_capacity = data['bearing_capacity']
soil_properties = data['density']
```

---

## 🤝 Contributing

### Development Setup
```bash
git clone https://github.com/kilickursat/SmolVLM-GeoEye.git
cd SmolVLM-GeoEye
pip install -r requirements.txt
cp .env.example .env
```

### Testing Enhanced Features
```bash
# Test data extraction
python -c "from app import EnhancedGeotechnicalDataExtractor; print('✅ Enhanced extraction ready')"

# Test monitoring
python enhanced_monitor_runpod.py --once

# Test visualization
streamlit run app.py
```

---

## 📜 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Repository**: [GitHub](https://github.com/kilickursat/SmolVLM-GeoEye)
- **RunPod Platform**: [RunPod.io](https://runpod.io)
- **SmolVLM Model**: [HuggingFace](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- **Issues**: [GitHub Issues](https://github.com/kilickursat/SmolVLM-GeoEye/issues)

---

## 📊 Version History

### v3.1.0 - Enhanced Resolution Update
- ✅ Fixed numerical data visualization issues
- ✅ Implemented real-time RunPod cost monitoring
- ✅ Enhanced SmolVLM usage visibility
- 🔧 Added advanced performance analytics
- 💰 Integrated cost optimization recommendations
- 📈 Enhanced interactive visualizations

### v3.0.0 - Initial Release
- 🤖 SmolVLM integration with RunPod
- 📄 Multi-format document processing
- 🏗️ Geotechnical data extraction
- 📊 Basic visualization capabilities

---

*Built with ❤️ for the geotechnical engineering community*
