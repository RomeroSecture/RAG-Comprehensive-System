import pytest
from fastapi.testclient import TestClient

from src.presentation.api.main import app


client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self):
        """Test the main health check endpoint."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "services" in data
        
        # Check service statuses
        assert "api" in data["services"]
        assert "database" in data["services"]
        assert "vector_store" in data["services"]
        assert "embedding_service" in data["services"]
    
    def test_liveness_probe(self):
        """Test Kubernetes liveness probe endpoint."""
        response = client.get("/api/v1/health/liveness")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "alive"
        assert "timestamp" in data
    
    def test_readiness_probe(self):
        """Test Kubernetes readiness probe endpoint."""
        response = client.get("/api/v1/health/readiness")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "ready"
        assert "timestamp" in data
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "RAG Comprehensive System API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "operational"
        assert data["docs"] == "/api/docs"
        assert data["health"] == "/api/v1/health"