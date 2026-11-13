"""
redis_client.py

Opens the connection to Redis (the whiteboard).
The login details come from the .env file, never from the code.
"""

import os

import redis
from dotenv import load_dotenv

# These must be filled in .env, or we cannot connect.
REQUIRED = ["REDIS_HOST", "REDIS_PASSWORD"]


def get_redis():
    # Read the .env file so its values become available.
    load_dotenv()

    # Stop early with a clear message if something is missing.
    missing = [name for name in REQUIRED if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")

    return redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        username=os.getenv("REDIS_USERNAME") or None,  # empty means none
        password=os.getenv("REDIS_PASSWORD"),
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
        decode_responses=True,  # give us normal text, not raw bytes
    )


# Quick test. Only runs when this file is run directly.
if __name__ == "__main__":
    client = get_redis()

    # 1. Knock on the door. It should answer True.
    print("Ping:", client.ping())

    # 2. Write one practice note that deletes itself after 60 seconds.
    test_key = "cost_tracker:test:hello"
    client.set(test_key, "it works", ex=60)

    # 3. Read it back.
    print("Read back:", client.get(test_key))

    # 4. Clean up our own practice note.
    client.delete(test_key)
    print("Cleaned up:", client.get(test_key) is None)