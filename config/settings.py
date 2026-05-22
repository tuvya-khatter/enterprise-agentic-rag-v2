"""Centralized configuration via Pydantic Settings."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # AWS / Bedrock
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    bedrock_generation_model: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    bedrock_embedding_model: str = "amazon.titan-embed-text-v2:0"

    # Database
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # Auth
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # Observability
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Optional
    cohere_api_key: str = ""

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit_per_minute: int = 60
    max_query_tokens: int = 4000


@lru_cache
def get_settings() -> Settings:
    return Settings()
