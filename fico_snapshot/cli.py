"""Command-line interface for creating FICO snapshots."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from fico_snapshot.snapshot import SnapshotError, create_fico_snapshot

FICO_CONSUMER_COLLECTION = "fico_consumer_data"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find a MongoDB user by BVN and save their FICO data as JSON."
    )
    parser.add_argument("bvn", help="BVN of the user to look up")
    args = parser.parse_args()

    env_file = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_file)

    mongo_uri = os.environ.get("MONGODB_URI")
    database_name = os.environ.get("MONGODB_DATABASE")
    collection_name = os.environ.get("MONGODB_COLLECTION")
    missing_settings = [
        name
        for name, value in (
            ("MONGODB_URI", mongo_uri),
            ("MONGODB_DATABASE", database_name),
            ("MONGODB_COLLECTION", collection_name),
        )
        if not value
    ]
    if missing_settings:
        print(
            f"Missing required environment variable(s): {', '.join(missing_settings)}",
            file=sys.stderr,
        )
        return 2

    try:
        with MongoClient(mongo_uri, serverSelectionTimeoutMS=5000) as client:
            database = client[database_name]
            collection = database[collection_name]
            destination_collection = database[FICO_CONSUMER_COLLECTION]
            snapshot_path = create_fico_snapshot(
                collection,
                destination_collection,
                args.bvn,
            )
    except (PyMongoError, SnapshotError, OSError, ValueError) as exc:
        print(f"Could not create FICO snapshot: {exc}", file=sys.stderr)
        return 1

    print(snapshot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
