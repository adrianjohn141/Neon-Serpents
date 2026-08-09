#!/bin/sh
set -eu
rm -rf /staging/current
mkdir -p /staging/current/minio
pg_dump -h postgres -U neon -d neon_serpents -Fc -f /staging/current/postgres.dump
mc alias set local http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
mc mirror --overwrite local/neon-serpents /staging/current/minio
restic snapshots >/dev/null 2>&1 || restic init
restic backup /staging/current --tag neon-serpents
restic forget --keep-daily 7 --keep-weekly 4 --keep-monthly 6 --prune
rm -rf /staging/current
