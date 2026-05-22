"""JWT issue/verify round-trip."""
import pytest
from fastapi import HTTPException

from src.auth.jwt import create_access_token, decode_token


def test_jwt_roundtrip():
    token = create_access_token("user-1", "tenant-a", "admin")
    payload = decode_token(token)
    assert payload["sub"] == "user-1"
    assert payload["tenant_id"] == "tenant-a"
    assert payload["role"] == "admin"


def test_invalid_token_raises():
    with pytest.raises(HTTPException):
        decode_token("not-a-valid-token")
