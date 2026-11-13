"""
db.py

Opens the connection to PostgreSQL.
The login details are read from the .env file,
so the password is never written in the code.
"""

import os

import psycopg
from dotenv import load_dotenv

# These must be filled in .env, or we cannot connect.
REQUIRED = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]


def get_connection():
    # Read the .env file so its values become available to the program.
    load_dotenv()

    # Stop early with a clear message if something is missing.
    missing = [name for name in REQUIRED if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")

    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),  # 5432 if not set
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode=os.getenv("DB_SSLMODE"),  # optional, ignored if empty
    )


# Quick test. Only runs when this file is run directly.
if __name__ == "__main__":
    with get_connection() as connection:
        with connection.cursor() as cursor:
            # Ask the database who we are and where we are.
            cursor.execute("SELECT current_database(), current_user")
            print("Connected to:", cursor.fetchone())

            # Count the lines in the big book. It should be 0.
            cursor.execute("SELECT count(*) FROM cost_ledger")
            print("Rows in cost_ledger:", cursor.fetchone()[0])