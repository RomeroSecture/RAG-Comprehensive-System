from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class HealthStatus(BaseModel):
    """Health status of a service component."""
    
    status: str = Field(..., description="Status of the component (healthy/unhealthy)")
    message: Optional[str] = Field(None, description="Additional status message")
    latency_ms: Optional[float] = Field(None, description="Latency in milliseconds")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "message": "Connection established",
                "latency_ms": 5.2
            }
        }
    )


class HealthResponse(BaseModel):
    """Overall health check response."""
    
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, HealthStatus] = Field(default_factory=dict, description="Individual service statuses")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T00:00:00Z",
                "version": "1.0.0",
                "services": {
                    "database": {
                        "status": "healthy",
                        "latency_ms": 2.5
                    },
                    "vector_store": {
                        "status": "healthy",
                        "latency_ms": 8.3
                    },
                    "embedding_service": {
                        "status": "healthy",
                        "message": "Model loaded"
                    }
                }
            }
        }
    )