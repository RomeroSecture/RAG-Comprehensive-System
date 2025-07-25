from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from src.shared.config.settings import get_settings
from .v1 import api_router
from .middleware.error_handler import error_handler_middleware
from .middleware.logging import logging_middleware


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    print("🚀 Starting RAG Comprehensive System API...")
    
    # Initialize connections, load models, etc.
    # This is where you'd initialize:
    # - Database connections
    # - Vector store connections
    # - Load embedding models
    # - Initialize services
    
    yield
    
    # Shutdown
    print("🔌 Shutting down RAG Comprehensive System API...")
    # Clean up resources


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="RAG Comprehensive System API",
        description="Advanced Retrieval-Augmented Generation System with Self-RAG, GraphRAG, and Multimodal capabilities",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    app.middleware("http")(error_handler_middleware)
    app.middleware("http")(logging_middleware)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=settings.security.cors_allow_credentials,
        allow_methods=settings.security.cors_allow_methods,
        allow_headers=settings.security.cors_allow_headers,
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Include routers
    app.include_router(api_router, prefix="/api/v1")
    
    # Root endpoint
    @app.get("/", response_class=JSONResponse)
    async def root() -> Dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "name": "RAG Comprehensive System API",
            "version": "1.0.0",
            "status": "operational",
            "docs": "/api/docs",
            "health": "/api/v1/health"
        }
    
    return app


# Create the application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.presentation.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )