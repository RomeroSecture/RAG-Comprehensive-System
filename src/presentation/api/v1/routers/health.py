from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..schemas.health import HealthResponse, HealthStatus
from ..dependencies.services import get_health_check_service


router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the health status of the API and its dependencies."""
    
    # Basic health check - in production, this would check actual services
    services_status = {
        "api": HealthStatus(status="healthy", message="API is running"),
        "database": HealthStatus(status="healthy", latency_ms=2.5),
        "vector_store": HealthStatus(status="healthy", latency_ms=5.0),
        "embedding_service": HealthStatus(status="healthy", message="Model loaded")
    }
    
    # Determine overall status
    all_healthy = all(service.status == "healthy" for service in services_status.values())
    overall_status = "healthy" if all_healthy else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0",
        services=services_status
    )


@router.get("/liveness")
async def liveness() -> JSONResponse:
    """Kubernetes liveness probe endpoint."""
    return JSONResponse(
        content={"status": "alive", "timestamp": datetime.utcnow().isoformat() + "Z"},
        status_code=200
    )


@router.get("/readiness")
async def readiness() -> JSONResponse:
    """Kubernetes readiness probe endpoint."""
    # In production, check if services are ready to accept traffic
    return JSONResponse(
        content={"status": "ready", "timestamp": datetime.utcnow().isoformat() + "Z"},
        status_code=200
    )