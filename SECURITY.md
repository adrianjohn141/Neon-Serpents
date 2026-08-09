# Security policy

Neon Serpents is designed to run on one machine through Docker Compose. It is
not an internet-facing deployment by default.

## Security boundaries

- The gateway and MinIO console bind to `127.0.0.1`; PostgreSQL, Redis, the API,
  trainer, and actor services have no host ports.
- Administrator passwords are stored as Argon2 hashes. Sessions are server-side
  Redis records and use `HttpOnly` and `SameSite=Strict` cookies.
- Administrator-changing requests require a CSRF token. Login attempts are
  rate-limited.
- Browser gameplay may fetch only the signed production model manifest and
  production bundles. Administrative training, release, backup, and analytics
  routes require an administrator session.
- `.env.docker`, artifacts, checkpoints, database files, certificates, and key
  files are excluded from Git and Docker build contexts.

## Before sharing or hosting it

1. Run `ops/bootstrap-local.ps1` to create fresh local credentials. Never copy
   a real `.env.docker` into source control, screenshots, support tickets, or a
   Docker image.
2. Keep `ALLOW_MANUAL_PROMOTION=false`. It is a deliberate local testing escape
   hatch, not a way to declare a benchmark failure as a production-quality win.
3. Do not expose port 8193 or the MinIO console to the internet as-is. An
   internet deployment needs TLS, `COOKIE_SECURE=true`, a hardened reverse
   proxy, firewall rules, backups, and an independent deployment review.
4. Rotate every local secret if `.env.docker` is ever copied outside the
   machine.

## Current hardening follow-ups

- The API/trainer image currently runs as root. It should be moved to a
  non-root service account after volume ownership is tested.
- Container base-image tags and the backup image's MinIO client download should
  be pinned to immutable digests/checksums for stronger supply-chain control.
- Keep dependency updates and vulnerability scans part of regular maintenance.

## Reporting a vulnerability

Do not publish credentials or a reproducible exploit in a public issue. Use a
private GitHub security advisory for the repository once it is enabled, or
contact the repository owner privately with the affected version, impact, and
safe reproduction steps.
