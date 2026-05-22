"""JWT issue + verify + FastAPI dependency."""
from datetime import datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from config.settings import get_settings

settings = get_settings()
bearer_scheme = HTTPBearer()


def create_access_token(user_id: str, tenant_id: str, role: str = "user") -> str:
    expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiration_hours)
    payload = {"sub": user_id, "tenant_id": tenant_id, "role": role, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict[str, Any]:
    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub")
    tenant_id = payload.get("tenant_id")
    if not user_id or not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claims",
        )
    return {"user_id": user_id, "tenant_id": tenant_id, "role": payload.get("role", "user")}
