"""
Tests for the health check endpoint.

Verifies that GET /health returns a 200 status with the expected
schema fields, regardless of whether the classifiers are trained.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a FastAPI test client.

    Note: The test client triggers the lifespan (model loading).
    On first run this may download the SpeechBrain model (~80 MB).
    """
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client: TestClient):
        """Health endpoint should always return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client: TestClient):
        """Response should contain all expected fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "embedding_model_loaded" in data
        assert "gender_classifier_loaded" in data
        assert "age_classifier_loaded" in data
        assert "version" in data

    def test_health_embedding_model_loaded(self, client: TestClient):
        """After startup, the embedding model should be loaded."""
        response = client.get("/health")
        data = response.json()

        # The SpeechBrain model should load during lifespan startup
        assert data["embedding_model_loaded"] is True

    def test_health_status_ok_when_model_loaded(self, client: TestClient):
        """Status should be 'ok' when the embedding model is available."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "ok"

    def test_health_version_present(self, client: TestClient):
        """Version string should be non-empty."""
        response = client.get("/health")
        data = response.json()

        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0
