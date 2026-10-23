"""MongoDB lookup and JSON snapshot creation."""

from datetime import date, datetime, timezone
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from bson import json_util
from pymongo.collection import Collection


class SnapshotError(Exception):
    """Raised when a matching user's FICO data cannot be snapshotted."""


def create_fico_snapshot(
    collection: Collection[dict[str, Any]],
    destination_collection: Collection[dict[str, Any]],
    bvn: str,
    output_dir: Path = Path("snapshots"),
    snapshot_date: date | None = None,
) -> Path:
    """Look up a user by BVN and save their FICO data and snapshot metadata."""
    if not bvn.strip():
        raise ValueError("BVN must not be empty.")

    record = collection.find_one(
        {"bvn": bvn},
        {"_id": 0, "bvn": 1, "name": 1, "fico": 1, "firstCentral": 1},
    )
    if record is None:
        raise SnapshotError("No user was found for the supplied BVN.")

    fico = record.get("fico")
    if fico is None:
        raise SnapshotError("The matching user record has no FICO data.")

    first_central = record.get("firstCentral")
    if first_central is None:
        raise SnapshotError(
            "The matching user record has no firstCentral data. "
            "Verify the source database and collection."
        )

    pulled_at = datetime.now(timezone.utc)
    snapshot_date = snapshot_date or pulled_at.date()
    snapshot_datetime = datetime(
        snapshot_date.year,
        snapshot_date.month,
        snapshot_date.day,
        tzinfo=timezone.utc,
    )
    snapshot = {
        "bvn": record.get("bvn", bvn),
        "name": record.get("name"),
        "fico": fico,
        "firstCentral": first_central,
        "snapshot_date": snapshot_datetime,
        "pulled_at": pulled_at,
    }

    name = record.get("name")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name or "").strip(".-_")
    safe_name = safe_name or "user"
    snapshot_id = uuid4().hex

    snapshot_directory = output_dir / f"{safe_name}-{snapshot_id}"
    snapshot_directory.mkdir(parents=True, exist_ok=False)
    snapshot_path = snapshot_directory / f"{snapshot_date.isoformat()}.json"
    snapshot_path.write_text(
        json_util.dumps(snapshot, indent=2),
        encoding="utf-8",
    )
    destination_collection.insert_one(snapshot)
    return snapshot_path
