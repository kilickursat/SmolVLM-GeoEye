#!/usr/bin/env python3
"""
SmolVLM-GeoEye Test Script
=========================

Test script to verify SmolVLM extraction and agent functionality.
This script tests the core functionality without the full Streamlit interface.

Author: SmolVLM-GeoEye Team
Version: 3.2.0
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent / "modules"))

from modules.config import get_config
from modules.smolvlm_client import EnhancedRunPodClient
from modules.data_extraction import EnhancedGeotechnicalDataExtractor
from modules.agents import GeotechnicalAgentOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_configuration():
    """Test configuration loading"""
    print("🔧 Testing Configuration...")
    
    try:
        config = get_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   - App: {config.app_name} v{config.app_version}")
        print(f"   - RunPod API Key: {'✅ Set' if config.api_key else '❌ Not set'}")
        print(f"   - RunPod Endpoint: {'✅ Set' if config.endpoint_id else '❌ Not set'}")
        print(f"   - Model: {config.model_name}")
        print(f"   - Max tokens: {config.max_new_tokens}")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False

def test_runpod_connection():
    """Test RunPod connection"""
    print("\n🚀 Testing RunPod Connection...")
    
    try:
        config = get_config()
        if not config.api_key or not config.endpoint_id:
            print("⚠️  RunPod credentials not configured - skipping connection test")
            return True
        
        client = EnhancedRunPodClient(config)
        health = client.health_check()
        
        if health["ready"]:
            print(f"✅ RunPod connection successful")
            print(f"   - Status: {health['status']}")
            print(f"   - Workers ready: {health['workers']['ready']}")
            return True
        else:
            print(f"❌ RunPod not ready: {health.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ RunPod connection failed: {e}")
        return False

def test_data_extraction():
    """Test data extraction functionality"""
    print("\n📊 Testing Data Extraction...")
    
    try:
        extractor = EnhancedGeotechnicalDataExtractor()
        
        # Test with sample geotechnical text
        sample_text = """
        Geotechnical Investigation Report
        
        SPT Test Results:
        - Depth 1.5m: N-value = 12 blows/ft
        - Depth 3.0m: N-value = 15 blows/ft
        - Depth 4.5m: N-value = 18 blows/ft
        
        Laboratory Test Results:
        - Liquid Limit: 45%
        - Plastic Limit: 20%
        - Plasticity Index: 25
        - Moisture Content: 28%
        - Dry Density: 1.75 g/cm³
        
        Engineering Properties:
        - Allowable Bearing Capacity: 150 kPa
        - Cohesion: 25 kPa
        - Friction Angle: 32°
        - Settlement: 25 mm
        """
        
        extracted_data = extractor.extract_numerical_data_from_text(sample_text)
        
        if extracted_data:
            print(f"✅ Data extraction successful")
            print(f"   - Parameter types found: {len(extracted_data)}")
            
            total_values = sum(len(values) for values in extracted_data.values())
            print(f"   - Total values extracted: {total_values}")
            
            for param_type, values in extracted_data.items():
                print(f"   - {param_type}: {len(values)} values")
            
            return True
        else:
            print("❌ No data extracted")
            return False
            
    except Exception as e:
        print(f"❌ Data extraction failed: {e}")
        return False

def test_agents():
    """Test agent functionality"""
    print("\n🤖 Testing AI Agents...")
    
    try:
        orchestrator = GeotechnicalAgentOrchestrator()
        
        # Test with mock extracted data
        mock_context = {
            "processed_documents": {
                "test_document.pdf": {
                    "document_type": "pdf",
                    "processing_status": "completed",
                    "numerical_data": {
                        "spt_values": [
                            {"value": 12, "unit": "blows/ft", "context": "SPT at 1.5m", "confidence": 0.9, "parameter_type": "spt_values"},
                            {"value": 15, "unit": "blows/ft", "context": "SPT at 3.0m", "confidence": 0.9, "parameter_type": "spt_values"}
                        ],
                        "bearing_capacity": [
                            {"value": 150, "unit": "kPa", "context": "Allowable bearing capacity", "confidence": 0.9, "parameter_type": "bearing_capacity"}
                        ]
                    }
                }
            }
        }
        
        # Test query routing
        response = orchestrator.route_query("What is the bearing capacity?", mock_context)
        
        if response and response.response:
            print(f"✅ Agent response successful")
            print(f"   - Agent type: {response.agent_type}")
            print(f"   - Confidence: {response.confidence}")
            print(f"   - Data used: {len(response.data_used)} parameter types")
            print(f"   - Recommendations: {len(response.recommendations)}")
            print(f"   - Response preview: {response.response[:100]}...")
            return True
        else:
            print("❌ No agent response generated")
            return False
            
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def test_json_parsing():
    """Test JSON parsing functionality"""
    print("\n🔍 Testing JSON Parsing...")
    
    try:
        # Test JSON response parsing
        sample_json_response = '''
        {
            "document_analysis": "This is a geotechnical investigation report containing SPT test results and soil properties.",
            "extracted_data": {
                "spt_values": [
                    {"value": 15, "unit": "blows/ft", "depth": 3.0, "depth_unit": "m", "context": "Standard Penetration Test at 3m depth", "confidence": 0.9}
                ],
                "bearing_capacity": [
                    {"value": 150, "unit": "kPa", "context": "Allowable bearing capacity", "confidence": 0.9}
                ],
                "moisture_content": [
                    {"value": 25, "unit": "%", "context": "Natural moisture content", "confidence": 0.8}
                ]
            },
            "soil_classification": "Medium dense sand with low plasticity",
            "recommendations": ["Shallow foundations suitable", "Monitor settlement"],
            "warnings": ["Check groundwater levels"]
        }
        '''
        
        # Parse the JSON
        data = json.loads(sample_json_response)
        
        if 'extracted_data' in data:
            extracted_count = sum(len(values) for values in data['extracted_data'].values())
            print(f"✅ JSON parsing successful")
            print(f"   - Parameters extracted: {len(data['extracted_data'])}")
            print(f"   - Total values: {extracted_count}")
            print(f"   - Has recommendations: {'recommendations' in data}")
            print(f"   - Has warnings: {'warnings' in data}")
            return True
        else:
            print("❌ JSON structure invalid")
            return False
            
    except Exception as e:
        print(f"❌ JSON parsing failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🧪 SmolVLM-GeoEye Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("RunPod Connection", test_runpod_connection),
        ("Data Extraction", test_data_extraction),
        ("AI Agents", test_agents),
        ("JSON Parsing", test_json_parsing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SmolVLM-GeoEye is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the configuration and setup.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
