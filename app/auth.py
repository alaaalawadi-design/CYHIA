"""
Accounts, sessions and result ownership for the UI.

One SQLite file next to the results (OUT_DIR/users.db), so it lives on the same
bind mount and survives a restart or rebuild of the UI container. Standard
library only: scrypt for passwords, random tokens for sessions (only their
SHA-256 is stored, so a leaked database cannot be replayed as a cookie).

Every finished result is recorded against the user who submitted it; History
shows a user only their own rows. Results made before accounts existed have no
owner and are visible to nobody until assigned:

    python -m app.auth users
    python -m app.auth assign-unowned <username>

Admins (who may edit the workflow controls on /admin) are only ever made from
this CLI, never through sign-up:

    python -m app.auth create-admin <username> <password>   # create or reset + promote
    python -m app.auth set-admin <username> on|off
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

OUT_DIR = Path(os.environ.get("OUT_DIR", "/app/outputs"))
DB_PATH = Path(os.environ.get("AUTH_DB", OUT_DIR / "users.db"))

COOKIE = "cyhia_session"
SESSION_DAYS = 30
# Set SIGNUP_CODE to require an invite code on sign-up; ALLOW_SIGNUP=0 closes it.
SIGNUP_CODE = os.environ.get("SIGNUP_CODE", "")
ALLOW_SIGNUP = os.environ.get("ALLOW_SIGNUP", "1") != "0"

USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,32}$")
MIN_PASSWORD = 8

# Brute-force brake: this many failed sign-ins per username in the window -> 429.
MAX_FAILS, FAIL_WINDOW = 5, 600.0
_FAILS: dict[str, list[float]] = {}
_FAILS_LOCK = threading.Lock()


class AuthError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class User:
    id: int
    username: str
    is_admin: bool = False


def _db() -> sqlite3.Connection:
    # A connection per call: FastAPI runs sync endpoints on a thread pool.
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init() -> None:
    with _db() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                pw_hash  BLOB NOT NULL,
                salt     BLOB NOT NULL,
                created  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires    REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS results (
                service TEXT NOT NULL,
                name    TEXT NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created REAL NOT NULL,
                PRIMARY KEY (service, name)
            );
            """
        )
        # Databases made before the admin role existed lack the column.
        cols = {r[1] for r in con.execute("PRAGMA table_info(users)")}
        if "is_admin" not in cols:
            con.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        con.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))


# ── passwords ────────────────────────────────────────────────────────────────
def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode(), salt=salt, n=2**14, r=8, p=1, maxmem=64 << 20, dklen=32
    )


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _check_fails(username: str) -> None:
    now = time.time()
    with _FAILS_LOCK:
        recent = [t for t in _FAILS.get(username, []) if now - t < FAIL_WINDOW]
        _FAILS[username] = recent
        if len(recent) >= MAX_FAILS:
            raise AuthError(429, "Too many failed attempts. Try again in a few minutes.")


def _note_fail(username: str) -> None:
    with _FAILS_LOCK:
        _FAILS.setdefault(username, []).append(time.time())


# ── accounts ─────────────────────────────────────────────────────────────────
def signup(username: str, password: str, code: str = "") -> User:
    if not ALLOW_SIGNUP:
        raise AuthError(403, "Sign-up is closed. Ask an administrator for an account.")
    if SIGNUP_CODE and not hmac.compare_digest(code.strip(), SIGNUP_CODE):
        raise AuthError(403, "Invalid sign-up code.")
    username = username.strip().lower()
    if not USERNAME_RE.match(username):
        raise AuthError(
            400, "Username must be 3–32 characters: letters, digits, '.', '_' or '-'."
        )
    if len(password) < MIN_PASSWORD:
        raise AuthError(400, f"Password must be at least {MIN_PASSWORD} characters.")
    salt = secrets.token_bytes(16)
    try:
        with _db() as con:
            cur = con.execute(
                "INSERT INTO users (username, pw_hash, salt, created) VALUES (?,?,?,?)",
                (username, _hash(password, salt), salt, time.time()),
            )
    except sqlite3.IntegrityError:
        raise AuthError(409, "That username is taken.") from None
    return User(cur.lastrowid, username)


def login(username: str, password: str) -> User:
    username = username.strip().lower()
    _check_fails(username)
    with _db() as con:
        row = con.execute(
            "SELECT id, pw_hash, salt, is_admin FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    # Hash even for an unknown user, so timing does not reveal which names exist.
    salt = row[2] if row else b"\0" * 16
    ok = hmac.compare_digest(_hash(password, salt), row[1] if row else b"")
    if not (row and ok):
        _note_fail(username)
        raise AuthError(401, "Wrong username or password.")
    with _FAILS_LOCK:
        _FAILS.pop(username, None)
    return User(row[0], username, bool(row[3]))


def create_admin(username: str, password: str) -> User:
    """Create `username` as an admin, or reset its password and promote it."""
    username = username.strip().lower()
    if not USERNAME_RE.match(username):
        raise AuthError(400, "Invalid username.")
    if len(password) < MIN_PASSWORD:
        raise AuthError(400, f"Password must be at least {MIN_PASSWORD} characters.")
    salt = secrets.token_bytes(16)
    with _db() as con:
        con.execute(
            "INSERT INTO users (username, pw_hash, salt, created, is_admin) "
            "VALUES (?,?,?,?,1) ON CONFLICT(username) DO UPDATE SET "
            "pw_hash = excluded.pw_hash, salt = excluded.salt, is_admin = 1",
            (username, _hash(password, salt), salt, time.time()),
        )
        uid = con.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()[0]
        # A password reset must also sign out every existing session.
        con.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
    return User(uid, username, True)


# ── sessions ─────────────────────────────────────────────────────────────────
def new_session(user: User) -> str:
    token = secrets.token_urlsafe(32)
    with _db() as con:
        con.execute(
            "INSERT INTO sessions (token_hash, user_id, expires) VALUES (?,?,?)",
            (_token_hash(token), user.id, time.time() + SESSION_DAYS * 86400),
        )
    return token


def user_for(token: str | None) -> User | None:
    if not token:
        return None
    with _db() as con:
        row = con.execute(
            "SELECT u.id, u.username, u.is_admin FROM sessions s "
            "JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires > ?",
            (_token_hash(token), time.time()),
        ).fetchone()
    return User(row[0], row[1], bool(row[2])) if row else None


def end_session(token: str | None) -> None:
    if token:
        with _db() as con:
            con.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


# ── result ownership ─────────────────────────────────────────────────────────
def set_owner(service: str, name: str, user_id: int) -> None:
    with _db() as con:
        con.execute(
            "INSERT OR REPLACE INTO results (service, name, user_id, created) "
            "VALUES (?,?,?,?)",
            (service, name, user_id, time.time()),
        )


def remove_owner(user_id: int, service: str, name: str) -> bool:
    with _db() as con:
        cur = con.execute(
            "DELETE FROM results WHERE service = ? AND name = ? AND user_id = ?",
            (service, name, user_id),
        )
        return cur.rowcount > 0


def owned(user_id: int) -> set[tuple[str, str]]:
    with _db() as con:
        return set(
            con.execute(
                "SELECT service, name FROM results WHERE user_id = ?", (user_id,)
            ).fetchall()
        )


def owns(user_id: int, service: str, name: str) -> bool:
    with _db() as con:
        return (
            con.execute(
                "SELECT 1 FROM results WHERE service = ? AND name = ? AND user_id = ?",
                (service, name, user_id),
            ).fetchone()
            is not None
        )


# ── admin CLI ────────────────────────────────────────────────────────────────
def _cli(argv: list[str]) -> int:
    init()
    if argv[:1] == ["users"]:
        with _db() as con:
            for uid, name, adm, n in con.execute(
                "SELECT u.id, u.username, u.is_admin, COUNT(r.name) FROM users u "
                "LEFT JOIN results r ON r.user_id = u.id GROUP BY u.id ORDER BY u.id"
            ):
                print(f"{uid:>4}  {name:<32} {'admin' if adm else '     '}  {n} results")
        return 0
    if argv[:1] == ["create-admin"] and len(argv) == 3:
        try:
            create_admin(argv[1], argv[2])
        except AuthError as e:
            print(e, file=sys.stderr)
            return 1
        print(f"{argv[1].lower()} is an admin")
        return 0
    if argv[:1] == ["set-admin"] and len(argv) == 3 and argv[2] in ("on", "off"):
        with _db() as con:
            cur = con.execute(
                "UPDATE users SET is_admin = ? WHERE username = ?",
                (int(argv[2] == "on"), argv[1].lower()),
            )
        if not cur.rowcount:
            print(f"no such user: {argv[1]}", file=sys.stderr)
            return 1
        print(f"{argv[1].lower()}: admin {argv[2]}")
        return 0
    if argv[:1] == ["assign-unowned"] and len(argv) == 2:
        # Hand every result on disk that has no owner yet (made before accounts
        # existed) to one user, so it appears in their History.
        with _db() as con:
            row = con.execute(
                "SELECT id FROM users WHERE username = ?", (argv[1].lower(),)
            ).fetchone()
        if not row:
            print(f"no such user: {argv[1]}", file=sys.stderr)
            return 1
        n = 0
        for folder in sorted(p for p in OUT_DIR.iterdir() if p.is_dir()):
            if folder.name in ("uploads", "inputs", "incoming"):
                continue
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() == ".mp4":
                    with _db() as con:
                        cur = con.execute(
                            "INSERT OR IGNORE INTO results VALUES (?,?,?,?)",
                            (folder.name, f.name, row[0], f.stat().st_mtime),
                        )
                    n += cur.rowcount
        print(f"assigned {n} results to {argv[1]}")
        return 0
    print(
        "usage: python -m app.auth users | assign-unowned <username> | "
        "create-admin <username> <password> | set-admin <username> on|off",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
