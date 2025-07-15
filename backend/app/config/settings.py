import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Configuration
    API_TITLE: str = "Climate Change Analytics Platform"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "AI-powered social media analytics for climate change discussions"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    RELOAD: bool = True
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./climate_analytics.db"
    DATABASE_ECHO: bool = False
    
    # Security Configuration
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # CORS Configuration
    ALLOWED_ORIGINS: list = ["*"]
    ALLOWED_METHODS: list = ["*"]
    ALLOWED_HEADERS: list = ["*"]
    
    # File Upload Configuration
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: list = [".csv", ".json", ".xlsx"]
    UPLOAD_DIRECTORY: str = "data/uploads"
    
    # Model Configuration
    MODEL_DIRECTORY: str = "data/models"
    SENTIMENT_MODEL_PATH: str = "data/models/sentiment_model.pkl"
    ENGAGEMENT_MODEL_PATH: str = "data/models/engagement_model.pkl"
    TOPIC_MODEL_PATH: str = "data/models/topic_model.pkl"
    
    # Data Processing Configuration
    MIN_TEXT_LENGTH: int = 5
    MAX_TEXT_LENGTH: int = 10000
    DEFAULT_PREPROCESSING_LEVEL: str = "standard"
    
    # ML Model Configuration
    SENTIMENT_CONFIDENCE_THRESHOLD: float = 0.6
    TOPIC_MIN_PROBABILITY: float = 0.1
    ENGAGEMENT_PREDICTION_THRESHOLD: float = 0.5
    
    # Topic Modeling Configuration
    MAX_TOPICS: int = 10
    MIN_TOPICS: int = 2
    TOPIC_WORDS_DISPLAY: int = 5
    
    # Cache Configuration
    CACHE_TTL: int = 3600  # 1 hour
    ENABLE_CACHING: bool = True
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "logs/climate_analytics.log"
    
    # Analytics Configuration
    ENABLE_ANALYTICS: bool = True
    ANALYTICS_BATCH_SIZE: int = 100
    
    # Report Configuration
    REPORTS_DIRECTORY: str = "reports"
    MAX_REPORTS_HISTORY: int = 50
    
    # Rate Limiting Configuration
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour
    
    # Environment specific settings
    ENVIRONMENT: str = "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Ensure required directories exist
def create_directories():
    """Create required directories if they don't exist"""
    directories = [
        settings.UPLOAD_DIRECTORY,
        settings.MODEL_DIRECTORY,
        settings.REPORTS_DIRECTORY,
        "data/raw",
        "data/processed",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize directories
create_directories()
