import asyncio
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Generator
from uuid import uuid4

import pytest
import pytest_asyncio

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_document_data():
    """Provide sample document data for testing."""
    return {
        "content": "This is a sample document content for testing purposes. It contains enough text to be meaningful.",
        "metadata": {
            "author": "Test Author",
            "category": "Technology",
            "tags": ["ai", "machine-learning", "nlp"]
        },
        "source": "https://example.com/test-document.pdf",
        "file_type": "pdf",
        "language": "en"
    }


@pytest.fixture
def sample_query_data():
    """Provide sample query data for testing."""
    return {
        "text": "What are the latest developments in artificial intelligence and machine learning?",
        "user_id": uuid4(),
        "session_id": uuid4(),
        "metadata": {
            "source": "api",
            "client_version": "1.0.0"
        },
        "max_results": 10,
        "similarity_threshold": 0.75
    }


@pytest.fixture
def sample_embedding_data():
    """Provide sample embedding data for testing."""
    return {
        "vector": [random.random() for _ in range(768)],  # Standard BERT dimension
        "model": "text-embedding-ada-002",
        "dimensions": 768
    }


@pytest.fixture
def sample_chunk_data():
    """Provide sample document chunk data for testing."""
    content = "This is a sample chunk of text from a larger document. It represents a coherent section."
    return {
        "document_id": uuid4(),
        "content": content,
        "metadata": {
            "page": 5,
            "section": "Introduction"
        },
        "chunk_index": 0,
        "start_char": 0,
        "end_char": len(content)
    }


@pytest.fixture
def mock_datetime(monkeypatch):
    """Mock datetime for consistent testing."""
    fixed_datetime = datetime(2024, 1, 1, 12, 0, 0)
    
    class MockDateTime:
        @classmethod
        def utcnow(cls):
            return fixed_datetime
    
    monkeypatch.setattr("datetime.datetime", MockDateTime)
    return fixed_datetime


@pytest.fixture
def test_settings():
    """Provide test-specific settings."""
    return {
        "database_url": "postgresql://test:test@localhost:5432/test_rag",
        "redis_url": "redis://localhost:6379/0",
        "embedding_model": "text-embedding-ada-002",
        "max_chunk_size": 1000,
        "chunk_overlap": 200,
        "vector_dimensions": 768,
        "test_mode": True
    }


@pytest.fixture
def clean_test_data(request):
    """Ensure test data is cleaned up after tests."""
    created_files = []
    
    def register_file(filepath):
        created_files.append(filepath)
    
    yield register_file
    
    # Cleanup
    for filepath in created_files:
        if os.path.exists(filepath):
            os.remove(filepath)


@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests."""
    import time
    
    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}
        
        def start(self, operation: str):
            self.metrics[operation] = {"start": time.time()}
        
        def end(self, operation: str):
            if operation in self.metrics:
                self.metrics[operation]["end"] = time.time()
                self.metrics[operation]["duration"] = (
                    self.metrics[operation]["end"] - 
                    self.metrics[operation]["start"]
                )
        
        def get_duration(self, operation: str) -> float:
            return self.metrics.get(operation, {}).get("duration", 0.0)
        
        def get_all_metrics(self) -> dict:
            return self.metrics
    
    return PerformanceTracker()


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add performance marker for benchmark tests
        if "benchmark" in item.name or "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)