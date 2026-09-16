"""Regression tests for SSRF hardening on model endpoint routes.

Verifies that POST /model-endpoints/test and POST /model-endpoints reject
user-supplied URLs that resolve to link-local (cloud metadata), non-HTTP
schemes, or (when MODELENDPOINT_BLOCK_PRIVATE_IPS=true) private/loopback
addresses — matching the existing SSRF guard pattern in embedding_routes.py.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def client(monkeypatch):
    """Build a TestClient with auth disabled and minimal DB stubs."""
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")

    # Stub heavy deps that model_routes imports at module level
    for mod in [
        "src.tls_overrides",
        "src.llm_core",
        "src.settings",
        "src.constants",
        "core.log_safety",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    # Provide real endpoint_resolver so the route logic works
    if "src.endpoint_resolver" not in sys.modules or isinstance(
        sys.modules["src.endpoint_resolver"], MagicMock
    ):
        mod = sys.modules.setdefault("src.endpoint_resolver", MagicMock())
        mod.normalize_base = lambda url: url
        mod.resolve_url = lambda url: url
        mod.build_chat_url = lambda base, **kw: base
        mod.build_models_url = lambda base, **kw: base + "/v1/models"
        mod.build_headers = lambda base, **kw: {}

    # Clear cached model_routes module so it picks up our stubs
    monkeypatch.delitem(sys.modules, "routes.model_routes", raising=False)

    from fastapi.testclient import TestClient
    import routes.model_routes as mr
    from fastapi import FastAPI

    app = FastAPI()
    router = mr.setup_model_routes(model_discovery=None)
    app.include_router(router)
    app.state.auth_enabled = False
    return TestClient(app, raise_server_exceptions=False)


class TestModelEndpointSSRF:
    """SSRF protection for POST /model-endpoints/test."""

    def test_cloud_metadata_blocked(self, client):
        """Cloud metadata IP (169.254.169.254) must be rejected."""
        resp = client.post(
            "/api/model-endpoints/test",
            data={"base_url": "http://169.254.169.254/latest/meta-data/"},
        )
        assert resp.status_code == 400
        assert "Rejected" in resp.json()["detail"]

    def test_non_http_scheme_blocked(self, client):
        """Non-HTTP schemes (file://, ftp://) must be rejected."""
        resp = client.post(
            "/api/model-endpoints/test",
            data={"base_url": "file:///etc/passwd"},
        )
        assert resp.status_code == 400
        assert "Rejected" in resp.json()["detail"]

    def test_loopback_always_rejected(self, client):
        """Loopback addresses are always rejected (not just in strict mode)."""
        resp = client.post(
            "/api/model-endpoints/test",
            data={"base_url": "http://localhost:11434/v1"},
        )
        assert resp.status_code == 400
        assert "Rejected" in resp.json()["detail"]

    def test_strict_mode_blocks_loopback(self, client):
        """MODELENDPOINT_BLOCK_PRIVATE_IPS=true blocks loopback."""
        os.environ["MODELENDPOINT_BLOCK_PRIVATE_IPS"] = "true"
        try:
            resp = client.post(
                "/api/model-endpoints/test",
                data={"base_url": "http://127.0.0.1:11434/v1"},
            )
            assert resp.status_code == 400
            assert "Rejected" in resp.json()["detail"]
        finally:
            os.environ.pop("MODELENDPOINT_BLOCK_PRIVATE_IPS", None)


class TestModelEndpointCreateSSRF:
    """SSRF protection for POST /model-endpoints."""

    def test_cloud_metadata_blocked(self, client):
        """Cloud metadata IP must be rejected on endpoint creation."""
        resp = client.post(
            "/api/model-endpoints",
            data={"base_url": "http://169.254.169.254/latest/meta-data/"},
        )
        assert resp.status_code == 400
        assert "Rejected" in resp.json()["detail"]

    def test_non_http_scheme_blocked(self, client):
        """Non-HTTP schemes must be rejected on endpoint creation."""
        resp = client.post(
            "/api/model-endpoints",
            data={"base_url": "gopher://evil:6379/secret"},
        )
        assert resp.status_code == 400
        assert "Rejected" in resp.json()["detail"]

    def test_loopback_always_rejected(self, client):
        """Loopback addresses are always rejected (not just in strict mode)."""
        resp = client.post(
            "/api/model-endpoints",
            data={"base_url": "http://localhost:11434/v1"},
        )
        assert resp.status_code == 400
        assert "Rejected" in resp.json()["detail"]
