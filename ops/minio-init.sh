#!/bin/sh
set -eu
attempt=0
until mc alias set local http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  [ "$attempt" -lt 30 ] || { echo 'MinIO did not become ready.' >&2; exit 1; }
  sleep 2
done
mc mb --ignore-existing local/neon-serpents
mc anonymous set none local/neon-serpents
mc version enable local/neon-serpents
