from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(..., env="DATABASE_URL")
    pool_size: int = Field(5, env="DB_POOL_SIZE")
    max_overflow: int = Field(10, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(3600, env="DB_POOL_RECYCLE")
    echo: bool = Field(False, env="DB_ECHO")
    
    class Config:
        env_prefix = "DATABASE_"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field("redis://localhost:6379", env="REDIS_URL")
    ttl: int = Field(3600, env="REDIS_TTL")
    max_connections: int = Field(50, env="REDIS_MAX_CONNECTIONS")
    decode_responses: bool = Field(True, env="REDIS_DECODE_RESPONSES")
    
    class Config:
        env_prefix = "REDIS_"


class OpenAISettings(BaseSettings):
    """OpenAI configuration settings."""
    
    api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = Field("text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")
    chat_model: str = Field("gpt-4", env="OPENAI_CHAT_MODEL")
    max_retries: int = Field(3, env="OPENAI_MAX_RETRIES")
    timeout: int = Field(60, env="OPENAI_TIMEOUT")
    temperature: float = Field(0.7, env="OPENAI_TEMPERATURE")
    max_tokens: int = Field(2000, env="OPENAI_MAX_TOKENS")
    
    class Config:
        env_prefix = "OPENAI_"


class AnthropicSettings(BaseSettings):
    """Anthropic configuration settings."""
    
    api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    model: str = Field("claude-3-opus-20240229", env="ANTHROPIC_MODEL")
    max_retries: int = Field(3, env="ANTHROPIC_MAX_RETRIES")
    timeout: int = Field(60, env="ANTHROPIC_TIMEOUT")
    
    class Config:
        env_prefix = "ANTHROPIC_"


class VectorStoreSettings(BaseSettings):
    """Vector store configuration settings."""
    
    type: str = Field("pgvector", env="VECTOR_STORE_TYPE")
    dimension: int = Field(1536, env="VECTOR_DIMENSION")
    index_type: str = Field("hnsw", env="VECTOR_INDEX_TYPE")
    index_params: dict = Field({"m": 16, "ef_construction": 200}, env="VECTOR_INDEX_PARAMS")
    search_params: dict = Field({"ef_search": 100}, env="VECTOR_SEARCH_PARAMS")
    
    # Qdrant settings
    qdrant_url: Optional[str] = Field(None, env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    qdrant_collection: str = Field("rag_documents", env="QDRANT_COLLECTION")
    
    class Config:
        env_prefix = "VECTOR_"


class DocumentProcessingSettings(BaseSettings):
    """Document processing configuration settings."""
    
    max_chunk_size: int = Field(1000, env="MAX_CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    min_chunk_size: int = Field(100, env="MIN_CHUNK_SIZE")
    ocr_enabled: bool = Field(True, env="OCR_ENABLED")
    supported_formats: List[str] = Field(
        ["pdf", "docx", "txt", "md", "html", "json", "csv", "xlsx"],
        env="SUPPORTED_FORMATS"
    )
    max_file_size_mb: int = Field(100, env="MAX_FILE_SIZE_MB")
    
    class Config:
        env_prefix = "DOC_"


class RetrievalSettings(BaseSettings):
    """Retrieval configuration settings."""
    
    default_strategy: str = Field("hybrid", env="DEFAULT_RETRIEVAL_STRATEGY")
    max_results: int = Field(10, env="MAX_RETRIEVAL_RESULTS")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")
    rerank_enabled: bool = Field(True, env="RERANK_ENABLED")
    rerank_model: str = Field("cross-encoder/ms-marco-MiniLM-L-12-v2", env="RERANK_MODEL")
    hybrid_alpha: float = Field(0.5, env="HYBRID_ALPHA")
    
    @field_validator("similarity_threshold")
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        return v
    
    class Config:
        env_prefix = "RETRIEVAL_"


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field("HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    cors_origins: List[str] = Field(["http://localhost:3000"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(["*"], env="CORS_ALLOW_HEADERS")
    
    class Config:
        env_prefix = "SECURITY_"


class MonitoringSettings(BaseSettings):
    """Monitoring configuration settings."""
    
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")
    sentry_environment: str = Field("development", env="SENTRY_ENVIRONMENT")
    otel_endpoint: Optional[str] = Field(None, env="OTEL_EXPORTER_OTLP_ENDPOINT")
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    
    class Config:
        env_prefix = "MONITORING_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application
    app_name: str = Field("RAG Comprehensive System", env="APP_NAME")
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    version: str = Field("0.1.0", env="APP_VERSION")
    
    # Sub-settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    openai: OpenAISettings = OpenAISettings()
    anthropic: AnthropicSettings = AnthropicSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    document_processing: DocumentProcessingSettings = DocumentProcessingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    security: SecuritySettings = SecuritySettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    # API settings
    api_v1_prefix: str = Field("/api/v1", env="API_V1_PREFIX")
    docs_url: str = Field("/docs", env="DOCS_URL")
    redoc_url: str = Field("/redoc", env="REDOC_URL")
    openapi_url: str = Field("/openapi.json", env="OPENAPI_URL")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(1000, env="RATE_LIMIT_PER_HOUR")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export commonly used settings
settings = get_settings()