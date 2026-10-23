from datetime import date, datetime, timezone
import json

import pytest

from fico_snapshot.snapshot import SnapshotError, create_fico_snapshot


class FakeCollection:
    def __init__(self, record):
        self.record = record
        self.query = None
        self.projection = None
        self.inserted = None

    def find_one(self, query, projection):
        self.query = query
        self.projection = projection
        return self.record

    def insert_one(self, document):
        self.inserted = document


def test_creates_snapshot_with_fico_and_metadata_and_inserts_it(tmp_path):
    fico = {
        "score": [{"source": "CRC", "data": {"score": 558}}],
        "lastUpdatedAt": datetime(2022, 5, 17, 10, 17, 43, tzinfo=timezone.utc),
    }
    first_central = {
        "lastUpdatedAt": datetime(2022, 11, 4, 9, 45, 6, tzinfo=timezone.utc),
        "score": {
            "totalNoOfDelinquentFacilities": 0,
            "totalNoOfLoans": 1,
            "totalNoOfActiveLoans": 0,
            "totalNoOfClosedLoans": 1,
            "totalNoOfInstitutions": 1,
            "totalOverdue": "0.00",
            "totalBorrowed": 0,
            "firstCentralEnquiryResultID": "25859585",
            "loanPerformance": False,
        },
    }
    source_collection = FakeCollection(
        {
            "bvn": "22231186083",
            "name": "Lionel Okeoghene Orishane",
            "fico": fico,
            "firstCentral": first_central,
        }
    )
    destination_collection = FakeCollection(None)

    path = create_fico_snapshot(
        source_collection,
        destination_collection,
        "22231186083",
        output_dir=tmp_path / "snapshots",
        snapshot_date=date(2026, 10, 21),
    )

    assert path.parent.parent == tmp_path / "snapshots"
    assert path.parent.name.startswith("Lionel-Okeoghene-Orishane-")
    assert path.name == "2026-10-21.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert snapshot["bvn"] == "22231186083"
    assert snapshot["name"] == "Lionel Okeoghene Orishane"
    assert snapshot["fico"] == {
        "score": [{"source": "CRC", "data": {"score": 558}}],
        "lastUpdatedAt": {"$date": "2022-05-17T10:17:43Z"},
    }
    assert snapshot["firstCentral"] == {
        "lastUpdatedAt": {"$date": "2022-11-04T09:45:06Z"},
        "score": {
            "totalNoOfDelinquentFacilities": 0,
            "totalNoOfLoans": 1,
            "totalNoOfActiveLoans": 0,
            "totalNoOfClosedLoans": 1,
            "totalNoOfInstitutions": 1,
            "totalOverdue": "0.00",
            "totalBorrowed": 0,
            "firstCentralEnquiryResultID": "25859585",
            "loanPerformance": False,
        },
    }
    assert snapshot["snapshot_date"] == {"$date": "2026-10-21T00:00:00Z"}
    assert snapshot["pulled_at"]["$date"].endswith("Z")
    assert destination_collection.inserted["bvn"] == "22231186083"
    assert destination_collection.inserted["name"] == "Lionel Okeoghene Orishane"
    assert destination_collection.inserted["fico"] == fico
    assert destination_collection.inserted["firstCentral"] == first_central
    assert destination_collection.inserted["snapshot_date"] == datetime(
        2026, 10, 21, tzinfo=timezone.utc
    )
    assert destination_collection.inserted["pulled_at"].tzinfo == timezone.utc
    assert source_collection.query == {"bvn": "22231186083"}
    assert source_collection.projection == {
        "_id": 0,
        "bvn": 1,
        "name": 1,
        "fico": 1,
        "firstCentral": 1,
    }


def test_missing_user_raises_snapshot_error(tmp_path):
    collection = FakeCollection(None)
    destination_collection = FakeCollection(None)

    with pytest.raises(SnapshotError, match="No user was found"):
        create_fico_snapshot(
            collection,
            destination_collection,
            "22231186083",
            output_dir=tmp_path,
        )


def test_user_without_fico_raises_snapshot_error(tmp_path):
    collection = FakeCollection({"name": "Example User", "fico": None})
    destination_collection = FakeCollection(None)

    with pytest.raises(SnapshotError, match="no FICO data"):
        create_fico_snapshot(
            collection,
            destination_collection,
            "22231186083",
            output_dir=tmp_path,
        )


def test_user_without_first_central_raises_snapshot_error(tmp_path):
    source_collection = FakeCollection(
        {
            "bvn": "22231186083",
            "name": "Example User",
            "fico": {"score": []},
        }
    )
    destination_collection = FakeCollection(None)

    with pytest.raises(SnapshotError, match="no firstCentral data"):
        create_fico_snapshot(
            source_collection,
            destination_collection,
            "22231186083",
            output_dir=tmp_path,
        )

    assert destination_collection.inserted is None


def test_empty_bvn_is_rejected_before_query(tmp_path):
    collection = FakeCollection(None)
    destination_collection = FakeCollection(None)

    with pytest.raises(ValueError, match="BVN must not be empty"):
        create_fico_snapshot(
            collection,
            destination_collection,
            "  ",
            output_dir=tmp_path,
        )

    assert collection.query is None
