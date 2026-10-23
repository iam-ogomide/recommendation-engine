import argparse
import json
import os
import re
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


PROJECT_DIR = Path(__file__).resolve().parent
SNAPSHOTS_DIR = PROJECT_DIR / "snapshots"


def iso_date(value: Any, field: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10]).isoformat()
        except ValueError as exc:
            raise ValueError(f"Invalid {field} date in subscriber record") from exc
    raise ValueError(f"Subscriber record is missing a valid {field} date")


def build_snapshot(
    subscriber: dict[str, Any], pulled_at: datetime
) -> dict[str, str]:
    required_fields = (
        "full_name",
        "subscriber_id",
        "bvn",
        "plan",
        "subscription_status",
        "next_run_date",
    )
    missing = [
        field
        for field in required_fields
        if field not in subscriber or subscriber[field] is None
    ]
    if missing:
        raise ValueError(
            "Subscriber record is missing required field(s): " + ", ".join(missing)
        )

    return {
        "name": str(subscriber["full_name"]),
        "subscriber_id": str(subscriber["subscriber_id"]),
        "bvn": str(subscriber["bvn"]),
        "plan": str(subscriber["plan"]),
        "status": str(subscriber["subscription_status"]),
        "cycle_date": iso_date(subscriber["next_run_date"], "next_run_date"),
        "snapshot_date": pulled_at.date().isoformat(),
        "pulled_at": pulled_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def snapshot_file_path(snapshot: dict[str, str]) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", snapshot["name"]).strip("-_")
    if not safe_name:
        safe_name = "subscriber"
    folder_name = f"{safe_name}-{uuid.uuid4().hex}"
    return SNAPSHOTS_DIR / folder_name / f"{snapshot['snapshot_date']}.json"


def main() -> int:
    load_dotenv(PROJECT_DIR / ".env")

    parser = argparse.ArgumentParser(
        description="Find a subscriber by BVN and save a JSON snapshot."
    )
    parser.add_argument("bvn", help="BVN of the subscriber to look up")
    args = parser.parse_args()
    bvn = args.bvn.strip()
    if not bvn:
        print("A non-empty BVN is required.", file=sys.stderr)
        return 1

    uri = os.getenv("MONGODB_URI")
    database_name = os.getenv("MONGODB_DATABASE")
    source_collection_name = os.getenv(
        "MONGODB_COLLECTION", "demo_subscribers_data"
    )
    snapshot_collection_name = os.getenv(
        "SNAPSHOT_COLLECTION", "demo_subscriber_snapshots"
    )
    missing = [
        name
        for name, value in (
            ("MONGODB_URI", uri),
            ("MONGODB_DATABASE", database_name),
        )
        if not value or "<" in value or ">" in value
    ]
    if missing:
        print(
            "Set the following values in your .env file: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    if source_collection_name == snapshot_collection_name:
        print(
            "MONGODB_COLLECTION and SNAPSHOT_COLLECTION must have different names.",
            file=sys.stderr,
        )
        return 1

    try:
        with MongoClient(uri, serverSelectionTimeoutMS=10000) as client:
            database = client[database_name]
            matches = list(
                database[source_collection_name]
                .find({"bvn": bvn})
                .limit(2)
            )
            if not matches:
                print("No subscriber found for that BVN.", file=sys.stderr)
                return 1
            if len(matches) > 1:
                print(
                    "More than one subscriber matched that BVN; no snapshot was created.",
                    file=sys.stderr,
                )
                return 1

            pulled_at = datetime.now(timezone.utc)
            snapshot = build_snapshot(matches[0], pulled_at)
            output_path = snapshot_file_path(snapshot)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            database[snapshot_collection_name].insert_one(snapshot)

        print(
            f"Snapshot saved to {output_path.relative_to(PROJECT_DIR)} and "
            f"{database_name}.{snapshot_collection_name}."
        )
        return 0
    except (OSError, ValueError, PyMongoError) as exc:
        print(f"Snapshot failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
