"""
Question 3 — Secure Login System (~40 min)
==========================================

Implement a simple in-memory user authentication system with:

1. register(username, password) → hashes the password and stores the user.
   - Raise ValueError if the username already exists.

2. login(username, password) → verifies credentials and returns a signed JWT token
   valid for 30 minutes.
   - Raise ValueError if username not found or password is wrong.

3. verify_token(token) → validates the JWT and returns the username if valid.
   - Raise ValueError if the token is expired or invalid.

Requirements:
- Never store plain-text passwords.
- Use bcrypt for hashing.
- Use PyJWT for tokens.

Install: pip install bcrypt PyJWT
"""

import bcrypt
import jwt
import datetime

SECRET_KEY = "super-secret-key"  # In production: load from env, never hardcode

users: dict[str, bytes] = {}  # username → bcrypt hash


def register(username: str, password: str) -> None:
    if username in users:
        raise ValueError(f"Username '{username}' already exists.")
    users[username] = bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def login(username: str, password: str) -> str:
    if username not in users or not bcrypt.checkpw(password.encode(), users[username]):
        raise ValueError("Invalid username or password.")
    payload = {
        "sub": username,
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def verify_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired.")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token.")


if __name__ == "__main__":
    register("alice", "password123")
    print("✅ Registered alice")

    try:
        register("alice", "other")
    except ValueError as e:
        print(f"✅ Duplicate blocked: {e}")

    try:
        login("alice", "wrong")
    except ValueError as e:
        print(f"✅ Wrong password blocked: {e}")

    token = login("alice", "password123")
    print(f"✅ Login OK — token: {token[:40]}...")
    print(f"✅ Token verified for: {verify_token(token)}")

    try:
        verify_token(token + "tampered")
    except ValueError as e:
        print(f"✅ Tampered token rejected: {e}")
