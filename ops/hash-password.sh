#!/bin/sh
set -eu
python -m pip install --quiet --disable-pip-version-check --root-user-action=ignore argon2-cffi
exec python /bootstrap/hash-password.py
