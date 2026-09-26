"""Tests for FastAPI Application Initialization, Middleware & CORS (Step 38)."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app, create_app


@pytest.fixture
def client():
    test_app = create_app()

    @test_app.get("/test-ping")
    def ping():
        return {"status": "ok"}

    @test_app.get("/test-error")
    def raise_err():
        raise RuntimeError("Simulated crash")

    return TestClient(test_app, raise_server_exceptions=False)


def test_app_request_id_and_timing_middleware(client):
    response = client.get("/test-ping")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"].startswith("req-")
    assert "X-Response-Time-Ms" in response.headers
    assert float(response.headers["X-Response-Time-Ms"]) >= 0.0


def test_custom_request_id_passthrough(client):
    custom_id = "my-custom-client-id-12345"
    response = client.get("/test-ping", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id


def test_cors_headers(client):
    response = client.options(
        "/test-ping",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ("*", "http://localhost:8501")


def test_global_exception_handler(client):
    response = client.get("/test-error")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert "request_id" in data
    assert "X-Request-ID" in response.headers
