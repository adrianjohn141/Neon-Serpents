from __future__ import annotations

import base64
import hashlib
import json

import boto3
from botocore.config import Config
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .config import Settings, get_settings


class ArtifactStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client = boto3.client(
            "s3",
            endpoint_url=self.settings.s3_endpoint,
            aws_access_key_id=self.settings.s3_access_key,
            aws_secret_access_key=self.settings.s3_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        checksum = hashlib.sha256(data).hexdigest()
        self.client.put_object(
            Bucket=self.settings.s3_bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            Metadata={"sha256": checksum},
        )
        return checksum

    def get_object(self, key: str) -> dict:
        return self.client.get_object(Bucket=self.settings.s3_bucket, Key=key)

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object under a validated experiment/release prefix."""
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.settings.s3_bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", []))
        deleted = 0
        for offset in range(0, len(keys), 1_000):
            batch = keys[offset:offset + 1_000]
            if batch:
                self.client.delete_objects(Bucket=self.settings.s3_bucket, Delete={"Objects": [{"Key": key} for key in batch]})
                deleted += len(batch)
        return deleted


def canonical_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sign_manifest(payload: dict, settings: Settings | None = None) -> str:
    configured = settings or get_settings()
    seed = base64.b64decode(configured.manifest_signing_seed)
    if len(seed) != 32:
        raise ValueError("MANIFEST_SIGNING_SEED must decode to exactly 32 bytes.")
    signature = Ed25519PrivateKey.from_private_bytes(seed).sign(canonical_json(payload))
    return base64.b64encode(signature).decode()


def manifest_public_key(settings: Settings | None = None) -> str:
    configured = settings or get_settings()
    seed = base64.b64decode(configured.manifest_signing_seed)
    if len(seed) != 32:
        raise ValueError("MANIFEST_SIGNING_SEED must decode to exactly 32 bytes.")
    public = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(public).decode()
