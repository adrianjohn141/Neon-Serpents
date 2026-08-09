from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    database_url: str = "sqlite:///./artifacts/local-training.db"
    redis_url: str = "redis://localhost:6379/0"
    s3_endpoint: str = "http://localhost:9000"
    s3_bucket: str = "neon-serpents"
    s3_access_key: str = "neon-local"
    s3_secret_key: str = ""
    admin_password_hash_b64: str = ""
    session_secret: str = ""
    manifest_signing_seed: str = ""
    server_public_origin: str = "http://localhost:8193"
    cookie_secure: bool = False
    allow_manual_promotion: bool = False
    session_ttl_seconds: int = 86_400
    trainer_cache_dir: Path = Path("/var/lib/neon-serpents/cache")
    trainer_grpc_bind: str = "0.0.0.0:50051"


@lru_cache
def get_settings() -> Settings:
    return Settings()
