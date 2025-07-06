#!/usr/bin/env python3
"""
SmolVLM-GeoEye Unit Tests
=========================

Unit tests for core modules to ensure production readiness.

Author: SmolVLM-GeoEye Team
Version: 3.1.0
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import modules to test
from modules.data_extraction import EnhancedGeotechnicalDataExtractor, ExtractedValue
from modules.config import Config, ProductionConfig
from modules.cache import InMemoryCache, CacheManager
from modules.agents import SoilAnalysisAgent, TunnelSupportAgent, SafetyChecklistAgent

# Test Data Extraction Module
class TestDataExtraction:
    def setup_method(self):
        self.extractor = EnhancedGeotechnicalDataExtractor()
    
    def test_extract_spt_values(self):
        """Test SPT value extraction"""
        text = """
        The SPT test results show N-values of 15 at 3.5m depth.
        Another test recorded SPT N=25 at 5.0m depth.
        The blow count was 30 at 7.5 meters.
        """
        
        result = self.extractor.extract_numerical_data_from_text(text)
        
        assert 'spt_values' in result
        assert len(result['spt_values']) >= 2
        
        # Check first value
        first_value = result['spt_values'][0]
        assert first_value.value == 15.0
        assert first_value.depth == 3.5
        
    def test_extract_bearing_capacity(self):
        """Test bearing capacity extraction"""
        text = """
        The allowable bearing capacity is 150 kPa.
        Ultimate bearing capacity = 450 kPa.
        Safe bearing pressure: 100 kPa
        """
        
        result = self.extractor.extract_numerical_data_from_text(text)
        
        assert 'bearing_capacity' in result
        assert len(result['bearing_capacity']) >= 3
        
        # Check values
        values = [v.value for v in result['bearing_capacity']]
        assert 150.0 in values
        assert 450.0 in values
        assert 100.0 in values
    
    def test_extract_from_dataframe(self):
        """Test extraction from structured data"""
        df = pd.DataFrame({
            'Depth (m)': [1.5, 3.0, 4.5, 6.0],
            'SPT N-Value': [10, 15, 20, 25],
            'Density (g/cm3)': [1.8, 1.9, 2.0, 2.1],
            'Moisture Content (%)': [12.5, 13.0, 14.2, 15.1]
        })
        
        result = self.extractor.extract_from_structured_data(df)
        
        assert 'spt_values' in result
        assert 'density' in result
        assert 'moisture_content' in result
        
        # Check SPT values have associated depths
        for spt in result['spt_values']:
            assert spt.depth is not None
            assert spt.depth_unit == 'm'
    
    def test_statistical_summary(self):
        """Test statistical summary generation"""
        data = {
            'spt_values': [
                ExtractedValue(value=10, unit='', context='', confidence=0.9, parameter_type='spt_values'),
                ExtractedValue(value=20, unit='', context='', confidence=0.9, parameter_type='spt_values'),
                ExtractedValue(value=30, unit='', context='', confidence=0.9, parameter_type='spt_values'),
            ]
        }
        
        summary = self.extractor.get_statistical_summary(data)
        
        assert 'spt_values' in summary
        stats = summary['spt_values']
        assert stats['count'] == 3
        assert stats['min'] == 10
        assert stats['max'] == 30
        assert stats['mean'] == 20
        assert stats['median'] == 20

# Test Configuration Module
class TestConfiguration:
    def test_basic_config(self):
        """Test basic configuration"""
        config = Config()
        
        assert config.app_name == "SmolVLM-GeoEye"
        assert config.app_version == "3.1.0"
        assert config.max_file_size_mb == 50
        assert config.cache_ttl_seconds == 3600
    
    def test_production_config(self):
        """Test production configuration"""
        with patch.dict(os.environ, {'SECRET_KEY': 'test_secret'}):
            config = ProductionConfig()
            
            assert config.debug == False
            assert config.log_level == "INFO"
            assert config.enable_auth == True
            assert config.secret_key == "test_secret"
            assert config.rate_limit_enabled == True
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = Config()
        config.cost_alert_threshold = -10
        
        with pytest.raises(ValueError):
            config._validate()

# Test Cache Module
class TestCache:
    def test_inmemory_cache(self):
        """Test in-memory cache operations"""
        cache = InMemoryCache()
        
        # Test set and get
        cache.set("test_key", "test_value", ttl=60)
        assert cache.get("test_key") == "test_value"
        
        # Test exists
        assert cache.exists("test_key") == True
        assert cache.exists("non_existent") == False
        
        # Test delete
        cache.delete("test_key")
        assert cache.get("test_key") is None
        
        # Test clear
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_cache_ttl(self):
        """Test cache TTL expiration"""
        cache = InMemoryCache()
        
        # Set with very short TTL
        cache.set("expire_key", "value", ttl=0.1)
        assert cache.get("expire_key") == "value"
        
        # Wait for expiration
        import time
        time.sleep(0.2)
        assert cache.get("expire_key") is None
    
    def test_cache_manager(self):
        """Test cache manager"""
        config = Config()
        config.cache_enabled = True
        cache_manager = CacheManager(config)
        
        # Test basic operations
        cache_manager.set("test", "value")
        assert cache_manager.get("test") == "value"
        
        # Test cache key generation
        key = cache_manager.make_key("func", 1, 2, param="value")
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length

# Test Agent Module
class TestAgents:
    def test_soil_analysis_agent(self):
        """Test soil analysis agent"""
        agent = SoilAnalysisAgent()
        
        # Test expertise detection
        assert agent.can_handle("What are the SPT values?") > 0
        assert agent.can_handle("Analyze the soil properties") > 0
        assert agent.can_handle("What is the weather?") == 0
        
        # Test analysis with mock data
        context = {
            'processed_documents': {
                'test.pdf': {
                    'numerical_data': {
                        'spt_values': [
                            {'value': 10, 'unit': '', 'depth': 2.0},
                            {'value': 15, 'unit': '', 'depth': 4.0},
                            {'value': 20, 'unit': '', 'depth': 6.0}
                        ]
                    }
                }
            }
        }
        
        response = agent.analyze("Analyze the SPT data", context)
        
        assert response.agent_type == "Soil Analysis Expert"
        assert response.confidence > 0.5
        assert len(response.response) > 0
        assert isinstance(response.recommendations, list)
        assert isinstance(response.warnings, list)
    
    def test_tunnel_support_agent(self):
        """Test tunnel support agent"""
        agent = TunnelSupportAgent()
        
        # Test expertise detection
        assert agent.can_handle("What tunnel support is needed?") > 0
        assert agent.can_handle("Analyze rock quality") > 0
        assert agent.can_handle("What is the soil type?") == 0
        
        # Test analysis with RQD data
        context = {
            'processed_documents': {
                'test.pdf': {
                    'numerical_data': {
                        'rqd': [
                            {'value': 45, 'unit': '%'},
                            {'value': 55, 'unit': '%'},
                            {'value': 60, 'unit': '%'}
                        ]
                    }
                }
            }
        }
        
        response = agent.analyze("Recommend tunnel support", context)
        
        assert response.agent_type == "Tunnel Engineering Expert"
        assert len(response.recommendations) > 0
    
    def test_safety_checklist_agent(self):
        """Test safety checklist agent"""
        agent = SafetyChecklistAgent()
        
        # Test expertise detection
        assert agent.can_handle("Generate safety checklist") > 0
        assert agent.can_handle("What are the risks?") > 0
        
        # Test checklist generation
        context = {'processed_documents': {}}
        response = agent.analyze("Generate safety checklist", context)
        
        assert response.agent_type == "Safety & Risk Assessment Expert"
        assert "checklist" in response.response.lower()
        assert len(response.recommendations) > 0

# Test RunPod Client Mock
class TestRunPodClient:
    @patch('requests.get')
    @patch('requests.post')
    def test_health_check(self, mock_post, mock_get):
        """Test RunPod client health check"""
        from modules.smolvlm_client import EnhancedRunPodClient
        
        # Mock health response
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            'workers': {'ready': 2, 'running': 1, 'idle': 0},
            'jobs': {'pending': 0, 'completed': 10, 'failed': 0}
        }
        
        config = Config()
        config.api_key = "test_key"
        config.endpoint_id = "test_endpoint"
        
        client = EnhancedRunPodClient(config)
        health = client.health_check()
        
        assert health['status'] == 'healthy'
        assert health['ready'] == True
        assert health['workers']['ready'] == 2
    
    @patch('requests.post')
    def test_run_sync(self, mock_post):
        """Test synchronous inference"""
        from modules.smolvlm_client import EnhancedRunPodClient
        
        # Mock successful response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'output': {
                'response': 'Test analysis result',
                'processing_time': '2.5s'
            }
        }
        
        config = Config()
        config.api_key = "test_key"
        config.endpoint_id = "test_endpoint"
        
        client = EnhancedRunPodClient(config)
        result = client.run_sync({'test': 'data'})
        
        assert result['status'] == 'success'
        assert 'output' in result
        assert result['output']['response'] == 'Test analysis result'

# Test Database Operations Mock
class TestDatabase:
    def test_document_operations(self):
        """Test document database operations"""
        # This would use an in-memory SQLite database for testing
        from modules.database import DatabaseManager
        
        db = DatabaseManager("sqlite:///:memory:")
        
        # Test save document
        doc_id = db.save_document(
            filename="test.pdf",
            document_type="pdf",
            file_size=1024,
            file_hash="abc123"
        )
        
        assert isinstance(doc_id, int)
        assert doc_id > 0
        
        # Test update status
        db.update_document_status(doc_id, "completed", processing_time=2.5)
        
        # Test save numerical data
        numerical_data = {
            'spt_values': [
                ExtractedValue(value=15, unit='', context='test', confidence=0.9, parameter_type='spt_values')
            ]
        }
        db.save_numerical_data(doc_id, numerical_data)

# Test Integration
@pytest.mark.integration
class TestIntegration:
    def test_full_document_processing_flow(self):
        """Test complete document processing flow"""
        # This would test the full integration of components
        # but requires all services to be running
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
