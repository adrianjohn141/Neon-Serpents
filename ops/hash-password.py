from __future__ import annotations

import sys

from argon2 import PasswordHasher


password = sys.stdin.read().rstrip("\r\n")
if not password:
    raise SystemExit("Administrator password was empty.")
print(PasswordHasher().hash(password))
