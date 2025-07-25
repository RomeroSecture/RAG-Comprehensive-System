from fastapi import APIRouter

from .routers import health, documents, query

# Create the main API router
api_router = APIRouter()

# Include all routers
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(query.router)

# Export the router
__all__ = ["api_router"]