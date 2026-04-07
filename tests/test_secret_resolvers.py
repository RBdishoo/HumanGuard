"""
Tests for _resolve_master_key() and _resolve_export_key() in backend/app.py.

Each test clears the module-level cache before running so tests are fully isolated.
"""

import sys
import os
import json
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import backend.app as app_module


def _reset_resolvers():
    """Clear the module-level _master_key and _export_key caches."""
    app_module._reset_secret_caches()


# ---------------------------------------------------------------------------
# _resolve_master_key tests
# ---------------------------------------------------------------------------

def test_resolve_master_key_from_env_var(monkeypatch):
    """Fast path: HUMANGUARD_MASTER_KEY env var is returned without calling Secrets Manager."""
    _reset_resolvers()
    monkeypatch.setenv("HUMANGUARD_MASTER_KEY", "test-master-key")

    mock_boto3 = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = app_module._resolve_master_key()

    assert result == "test-master-key"
    mock_boto3.client.assert_not_called()
    _reset_resolvers()


def test_resolve_master_key_from_secrets_manager(monkeypatch):
    """Cold-start path: fetches key from Secrets Manager when env var is absent."""
    _reset_resolvers()
    monkeypatch.delenv("HUMANGUARD_MASTER_KEY", raising=False)
    monkeypatch.setenv("MASTER_KEY_SECRET_NAME", "humanGuard/masterKey")

    secret_payload = json.dumps({"key": "sm-master-key-value"})
    mock_sm = mock.MagicMock()
    mock_sm.get_secret_value.return_value = {"SecretString": secret_payload}
    mock_boto3 = mock.MagicMock()
    mock_boto3.client.return_value = mock_sm

    with mock.patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = app_module._resolve_master_key()

    assert result == "sm-master-key-value"
    mock_boto3.client.assert_called_once_with("secretsmanager", region_name=mock.ANY)
    mock_sm.get_secret_value.assert_called_once_with(SecretId="humanGuard/masterKey")
    _reset_resolvers()


# ---------------------------------------------------------------------------
# _resolve_export_key tests
# ---------------------------------------------------------------------------

def test_resolve_export_key_from_env_var(monkeypatch):
    """Fast path: EXPORT_API_KEY env var is returned without calling Secrets Manager."""
    _reset_resolvers()
    monkeypatch.setenv("EXPORT_API_KEY", "test-export-key")

    mock_boto3 = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = app_module._resolve_export_key()

    assert result == "test-export-key"
    mock_boto3.client.assert_not_called()
    _reset_resolvers()


def test_resolve_export_key_from_secrets_manager(monkeypatch):
    """Cold-start path: fetches key from Secrets Manager when env var is absent."""
    _reset_resolvers()
    monkeypatch.delenv("EXPORT_API_KEY", raising=False)
    monkeypatch.setenv("EXPORT_KEY_SECRET_NAME", "humanGuard/exportKey")

    secret_payload = json.dumps({"key": "sm-export-key-value"})
    mock_sm = mock.MagicMock()
    mock_sm.get_secret_value.return_value = {"SecretString": secret_payload}
    mock_boto3 = mock.MagicMock()
    mock_boto3.client.return_value = mock_sm

    with mock.patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = app_module._resolve_export_key()

    assert result == "sm-export-key-value"
    mock_boto3.client.assert_called_once_with("secretsmanager", region_name=mock.ANY)
    mock_sm.get_secret_value.assert_called_once_with(SecretId="humanGuard/exportKey")
    _reset_resolvers()
