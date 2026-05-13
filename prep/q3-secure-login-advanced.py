"""
Question 3 — Advanced Auth: JWT vs OAuth2/API Keys
===================================================

─── JWT vs OAuth2 — Key Difference ──────────────────────────────────────────

JWT (JSON Web Token)
  - A TOKEN FORMAT, not an auth protocol.
  - A signed, self-contained blob: header.payload.signature
  - The server encodes claims (who you are, expiry, role) into the token and
    signs it with a secret. Any server that knows the secret can verify it
    WITHOUT hitting a database — that's the key advantage.
  - Used for: API auth, session replacement, service-to-service auth.

  Flow:
    1. User logs in with username/password
    2. Server returns a signed JWT
    3. Client sends JWT in every request: Authorization: Bearer <token>
    4. Server verifies signature + expiry locally — no DB lookup needed

OAuth2 / API Keys
  - OAuth2 is an AUTHORIZATION PROTOCOL (not just a token format).
  - Designed for delegated access: "allow app X to act on my behalf on service Y"
    without giving app X my password.
  - For AI/API products you often issue your own API keys (simplified OAuth2 style):
    client sends key in header, you validate server-side via a DB lookup.

  Flow (API key style):
    1. Server generates a random key and gives it to the client (shown once)
    2. Server stores only the HASH of the key (never the key itself)
    3. Client sends key in every request: Authorization: Bearer sk-<key>
    4. Server hashes the incoming key and compares against stored hash

  Summary:
  ┌────────────┬──────────────────────────────┬──────────────────────────────┐
  │            │ JWT                          │ API Key (OAuth2-style)       │
  ├────────────┼──────────────────────────────┼──────────────────────────────┤
  │ What is it │ Token format                 │ Opaque random string         │
  │ Stateless  │ Yes — no DB lookup           │ No — requires DB lookup      │
  │ Expiry     │ Built-in (exp claim)         │ Manual (store expiry in DB)  │
  │ Revocable  │ Hard (need a blocklist)      │ Easy (delete from DB)        │
  │ Use case   │ User sessions, microservices │ Service-to-service, AI APIs  │
  └────────────┴──────────────────────────────┴──────────────────────────────┘

Install: pip install bcrypt PyJWT
"""

import secrets
import hashlib
import bcrypt
import jwt
import datetime

SECRET_KEY = "super-secret-key"  # In production: load from env


# ══════════════════════════════════════════════════════════════════════════════
# PART A — JWT with roles
# ══════════════════════════════════════════════════════════════════════════════

users: dict[str, dict] = {}  # username → {hash, role}


def register(username: str, password: str, role: str = "user") -> None:
    if username in users:
        raise ValueError(f"Username '{username}' already exists.")
    users[username] = {
        "hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt()),
        "role": role,
    }


def login(username: str, password: str) -> str:
    user = users.get(username)
    if not user or not bcrypt.checkpw(password.encode(), user["hash"]):
        raise ValueError("Invalid username or password.")
    payload = {
        "sub": username,
        "role": user["role"],
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def verify_token(token: str) -> dict:
    """Returns {"sub": username, "role": role}."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired.")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token.")


def require_role(token: str, role: str) -> str:
    """Verify token and enforce role. Returns username if authorized."""
    payload = verify_token(token)
    if payload["role"] != role:
        raise PermissionError(f"Requires role '{role}', got '{payload['role']}'.")
    return payload["sub"]


# ══════════════════════════════════════════════════════════════════════════════
# PART B — API Key Auth (OAuth2-style)
# ══════════════════════════════════════════════════════════════════════════════

api_keys: dict[str, dict] = {}  # sha256(key) → {owner, role}


def create_api_key(owner: str, role: str = "user") -> str:
    """Generate API key for a service. Returns plain key — shown only once."""
    plain_key = "sk-" + secrets.token_hex(32)
    hashed = hashlib.sha256(plain_key.encode()).hexdigest()
    api_keys[hashed] = {"owner": owner, "role": role}
    return plain_key


def validate_api_key(auth_header: str) -> dict:
    """Parse 'Bearer sk-...' header and validate. Returns {owner, role}."""
    if not auth_header.startswith("Bearer "):
        raise ValueError("Missing or malformed Authorization header.")
    plain_key = auth_header[len("Bearer "):]
    hashed = hashlib.sha256(plain_key.encode()).hexdigest()
    if hashed not in api_keys:
        raise ValueError("Invalid API key.")
    return api_keys[hashed]


def revoke_api_key(plain_key: str) -> None:
    """Revoke an API key — easy with DB lookup, impossible with pure JWT."""
    hashed = hashlib.sha256(plain_key.encode()).hexdigest()
    api_keys.pop(hashed, None)


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("── Part A: JWT with roles ───────────────────────")

    register("alice", "password123", role="user")
    register("bob", "adminpass", role="admin")

    user_token = login("alice", "password123")
    admin_token = login("bob", "adminpass")
    print(f"✅ alice token: {user_token[:40]}...")

    print(f"✅ Token payload: {verify_token(user_token)}")

    print(f"✅ Admin check passed for: {require_role(admin_token, 'admin')}")

    try:
        require_role(user_token, "admin")
    except PermissionError as e:
        print(f"✅ Role enforced: {e}")

    print("\n── Part B: API Key ──────────────────────────────")

    key = create_api_key("service-x", role="admin")
    print(f"✅ API key created: {key[:20]}...")

    info = validate_api_key(f"Bearer {key}")
    print(f"✅ Key valid — owner: {info['owner']}, role: {info['role']}")

    revoke_api_key(key)
    try:
        validate_api_key(f"Bearer {key}")
    except ValueError as e:
        print(f"✅ Revoked key rejected: {e}")

    try:
        validate_api_key("Bearer sk-fakekey")
    except ValueError as e:
        print(f"✅ Invalid key rejected: {e}")
