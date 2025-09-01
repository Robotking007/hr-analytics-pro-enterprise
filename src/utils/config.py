"""
Configuration management for HR Performance Analytics Pro
"""
import os
from typing import Dict, Any
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # Supabase Configuration
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    supabase_service_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")
    
    # Database Configuration
    database_url: str = os.getenv("DATABASE_URL", "")
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "hr-analytics-secret-key")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Application Settings
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_upload_size: str = os.getenv("MAX_UPLOAD_SIZE", "50MB")
    
    # Model Settings
    fairness_threshold: float = 0.8
    privacy_budget: float = 1.0
    model_retrain_threshold: float = 0.05
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()

def get_model_config() -> Dict[str, Any]:
    """Get model configuration"""
    return {
        "ensemble_models": ["random_forest", "xgboost", "lightgbm", "neural_network", "svm"],
        "hyperparameter_tuning": True,
        "cross_validation_folds": 5,
        "feature_selection_k": 50,
        "fairness_threshold": settings.fairness_threshold
    }

def get_database_config() -> Dict[str, Any]:
    """Get database configuration"""
    return {
        "supabase_url": settings.supabase_url,
        "supabase_key": settings.supabase_key,
        "connection_timeout": 30,
        "max_connections": 10
    }
