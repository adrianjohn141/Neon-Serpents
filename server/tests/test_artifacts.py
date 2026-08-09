from __future__ import annotations

import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from server.app.artifacts import canonical_json, sign_manifest
from server.app.config import Settings


def test_manifest_signature_covers_exact_canonical_payload() -> None:
    seed = bytes(range(32))
    settings = Settings(manifest_signing_seed=base64.b64encode(seed).decode())
    payload = {"z": 1.25, "a": {"snakeId": "nova"}}
    signature = base64.b64decode(sign_manifest(payload, settings))
    Ed25519PrivateKey.from_private_bytes(seed).public_key().verify(signature, canonical_json(payload))
