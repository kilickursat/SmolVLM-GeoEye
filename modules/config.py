#!/usr/bin/env python3
"""
SmolVLM-GeoEye Configuration Module
===================================

Configuration management for production deployment.
Handles environment variables, settings, and feature flags.

Author: SmolVLM-GeoEye Team
Version: 3.1.0
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Base configuration class"""
    # RunPod Settings
    api_key: str = field(default_factory=lambda: os.getenv("RUNPOD_API_KEY", ""))
    endpoint_id: str = field(default_factory=lambda: os.getenv("RUNPOD_ENDPOINT_ID", ""))
    runpod_timeout: int = field(default_factory=lambda: int(os.getenv("RUNPOD_TIMEOUT", "300")))
    runpod_max_retries: int = field(default_factory=lambda: int(os.getenv("RUNPOD_MAX_RETRIES", "3")))
    
    # HuggingFace Settings
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    
    # Application Settings
    app_name: str = "SmolVLM-GeoEye"
    app_version: str = "3.1.0"
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # Data Processing Settings
    max_file_size_mb: int = 50
    supported_image_formats: list = field(default_factory=lambda: ["png", "jpg", "jpeg", "gif", "bmp"])
    supported_document_formats: list = field(default_factory=lambda: ["pdf", "txt", "csv", "xlsx", "json"])
    
    # Model Settings
    model_name: str = "HuggingFaceTB/SmolVLM-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.3
    do_sample: bool = True
    top_p: float = 0.9
    
    # Database Settings
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///geotechnical_data.db"))
    
    # Cache Settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    
    # Monitoring Settings
    enable_metrics: bool = True
    metrics_port: int = 8000
    
    # Cost Management
    cost_tracking_enabled: bool = True
    cost_per_minute: float = 0.0013  # Default for RTX A6000
    cost_alert_threshold: float = field(default_factory=lambda: float(os.getenv("COST_ALERT_THRESHOLD", "100.0")))
    
    # Worker Settings
    max_workers: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "10")))
    min_workers: int = field(default_factory=lambda: int(os.getenv("MIN_WORKERS", "0")))
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.2
    
    # Health Check Settings
    health_check_interval: int = 30
    health_check_timeout: int = 10
    
    # Feature Flags
    enable_smol_agents: bool = True
    enable_visualization: bool = True
    enable_async_processing: bool = True
    enable_cost_optimization: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate()
        self._setup_logging()
        
    def _validate(self):
        """Validate configuration values"""
        if not self.api_key:
            logger.warning("RUNPOD_API_KEY not set - RunPod features will be disabled")
        
        if not self.endpoint_id:
            logger.warning("RUNPOD_ENDPOINT_ID not set - RunPod features will be disabled")
        
        if self.max_file_size_mb > 100:
            logger.warning(f"Large max_file_size_mb ({self.max_file_size_mb}MB) may impact performance")
        
        if self.cost_alert_threshold < 0:
            raise ValueError("cost_alert_threshold must be positive")
        
        if self.max_workers < self.min_workers:
            raise ValueError("max_workers must be >= min_workers")
    
    def _setup_logging(self):
        """Configure logging based on settings"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'app_name': self.app_name,
            'app_version': self.app_version,
            'debug': self.debug,
            'runpod_configured': bool(self.api_key and self.endpoint_id),
            'features': {
                'smol_agents': self.enable_smol_agents,
                'visualization': self.enable_visualization,
                'async_processing': self.enable_async_processing,
                'cost_optimization': self.enable_cost_optimization,
            },
            'limits': {
                'max_file_size_mb': self.max_file_size_mb,
                'max_workers': self.max_workers,
                'min_workers': self.min_workers,
            }
        }

@dataclass
class ProductionConfig(Config):
    """Production-specific configuration"""
    # Production overrides
    debug: bool = False
    log_level: str = "INFO"
    
    # Security Settings
    enable_auth: bool = True
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", ""))
    allowed_origins: list = field(default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "*").split(","))
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    # Backup Settings
    enable_backups: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    
    # Health Check Settings (inherited from base class)
    # health_check_interval: int = 30
    # health_check_timeout: int = 10
    
    def __post_init__(self):
        """Additional production validation"""
        super().__post_init__()
        
        if self.enable_auth and not self.secret_key:
            raise ValueError("SECRET_KEY must be set for production with authentication enabled")
        
        if self.rate_limit_requests < 10:
            logger.warning("Very low rate limit may impact user experience")

class ConfigManager:
    """Manages configuration loading and access"""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize configuration manager"""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()
        
        self._config_cache = {}
        
    def get_config(self, config_class=Config) -> Config:
        """Get configuration instance (cached)"""
        class_name = config_class.__name__
        
        if class_name not in self._config_cache:
            self._config_cache[class_name] = config_class()
        
        return self._config_cache[class_name]
    
    def reload_config(self):
        """Reload configuration from environment"""
        self._config_cache.clear()
        load_dotenv(override=True)
    
    def save_config_snapshot(self, filepath: str):
        """Save current configuration snapshot (non-sensitive values only)"""
        config = self.get_config()
        safe_config = {
            'app_name': config.app_name,
            'app_version': config.app_version,
            'debug': config.debug,
            'log_level': config.log_level,
            'features': {
                'smol_agents': config.enable_smol_agents,
                'visualization': config.enable_visualization,
                'async_processing': config.enable_async_processing,
                'cost_optimization': config.enable_cost_optimization,
            },
            'limits': {
                'max_file_size_mb': config.max_file_size_mb,
                'max_workers': config.max_workers,
                'min_workers': config.min_workers,
            },
            'model_settings': {
                'model_name': config.model_name,
                'max_new_tokens': config.max_new_tokens,
                'temperature': config.temperature,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(safe_config, f, indent=2)
        
        logger.info(f"Configuration snapshot saved to {filepath}")

# Singleton instance
_config_manager = ConfigManager()

def get_config(config_class=Config) -> Config:
    """Get configuration instance"""
    return _config_manager.get_config(config_class)

def reload_config():
    """Reload configuration"""
    _config_manager.reload_config()
