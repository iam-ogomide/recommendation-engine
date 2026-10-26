"""
Step 2 of the CreditCliq agentic runner
=======================================
Pull the latest bureau report for one subscriber and store a dated snapshot.

Rules:
  - Public interface takes subscriber_id. The BVN is looked up inside this function,
    used for the report call, and never stored in the snapshot or written to logs.
  - snapshot_date is the date WE pulled the report, not a date found inside it, so a
    stale report cannot back-date a snapshot. A stale report is flagged instead.
  - Bureau names are canonical everywhere: FIRST_CENTRAL, CRC, CREDIT_REGISTRY.
  - A missing non-performing count is NOT silently zero: it is flagged NPL_UNKNOWN.
  - On failure the orchestrator logs the reason; SnapshotError.retryable says whether a
    retry (next 3 days) makes sense.

The snapshot shape is core.BureauSnapshot, so build_context() consumes it directly.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from core import WAT, BureauSnapshot

FIRST_CENTRAL = "FIRST_CENTRAL"
CRC = "CRC"
CREDIT_REGISTRY = "CREDIT_REGISTRY"
_BUREAU_ALIASES = {
    "FIRSTCENTRAL": FIRST_CENTRAL,
    "CRC": CRC,
    "CREDITREGISTRY": CREDIT_REGISTRY,
}


def canonical_bureau(value: Any) -> str | None:
    """'First Central', 'FIRST_CENTRAL', 'firstcentral' -> 'FIRST_CENTRAL'. Unknown -> None."""
    key = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return _BUREAU_ALIASES.get(key)


def _norm_status(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


class SnapshotError(Exception):
    """A usable bureau snapshot could not be produced."""

    def __init__(self, reason: str, retryable: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


class SubscriberLookup(Protocol):
    def get_bvn(self, subscriber_id: str) -> str | None: ...


class ReportService(Protocol):
    """The existing bureau report service: BVN in, raw dict out."""

    def pull(self, bvn: str) -> dict[str, Any]: ...


class SnapshotStore(Protocol):
    def save(self, snapshot: BureauSnapshot) -> None: ...


class SnapshotReader(Protocol):
    def load_previous(self, subscriber_id: str, bureau: str, before: date) -> BureauSnapshot | None:
        """Latest snapshot for this subscriber and bureau dated STRICTLY before `before`."""
        ...


class Step2Config(BaseModel):
    max_retry_days: int = 3
    default_bureau: str = FIRST_CENTRAL  # used when the service does not say which bureau
    # PLACEHOLDER: confirm the real strings each bureau uses for a non-performing facility.
    non_performing_statuses: list[str] = Field(default_factory=lambda: ["NONPERFORMING"])
    count_closed_facilities: bool = True  # open question for the PM
    max_report_age_days: int = 45  # older than this gets the STALE_REPORT flag


# ---------------------------------------------------------------------------
# Extraction from the raw report
# ---------------------------------------------------------------------------

def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if re.fullmatch(r"-?\d+(\.0+)?", s):
            return int(float(s))
    return None

def _dig(obj: Any, *path: str) -> Any:
    """Safe nested lookup: _dig(raw, 'classic', 'score', 'NPL') -> value or None."""
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _for_bureau(items: Any, bureau: str) -> int | None:
    """List of {source, score | value | data:{score}} -> the number for this bureau."""
    if not isinstance(items, list):
        return None
    for entry in items:
        if not isinstance(entry, dict) or canonical_bureau(entry.get("source")) != bureau:
            continue
        data = entry.get("data")
        for candidate in (
            data.get("score") if isinstance(data, dict) else None,
            entry.get("score"),
            entry.get("value"),
        ):
            number = _to_int(candidate)
            if number is not None:
                return number
    return None

def _extract_bureau(raw: dict[str, Any], default: str) -> str:
    value = raw.get("bureau") or raw.get("bureau_name") or raw.get("source")
    if not value:
        return default
    bureau = canonical_bureau(value)
    if bureau is None:
        raise SnapshotError("UNKNOWN_BUREAU")
    return bureau


def _extract_score(raw: dict[str, Any], bureau: str) -> int | None:
    """Flat `score`, else this bureau's entry in the stored per-bureau lists."""
    flat = _to_int(raw.get("score"))
    if flat is not None:
        return flat
    for items in (
        _dig(raw, "fico", "score"),
        _dig(raw, "classic", "score", "ficoScores"),
        _dig(raw, "advanced", "score", "score", "ficoScores"),
    ):
        value = _for_bureau(items, bureau)
        if value is not None:
            return value
    return None
    
def _find_facilities(raw: dict[str, Any]) -> list | None:
    facilities = raw.get("loanPerformance")
    if isinstance(facilities, list):
        return facilities
    first_central = raw.get("firstCentral")
    if isinstance(first_central, dict):
        block = first_central.get("score")
        if isinstance(block, dict) and isinstance(block.get("loanPerformance"), list):
            return block["loanPerformance"]
    return None


def _extract_non_performing(raw: dict[str, Any], cfg: Step2Config, bureau: str) -> int | None:
    """Count of non-performing facilities, or None when it cannot be determined."""
    for key in ("non_performing_count", "non_performing_facilities", "npl_count"):
        value = _to_int(raw.get(key))
        if value is not None:
            return max(0, value)

    for items in (
        _dig(raw, "classic", "score", "NPL"),
        _dig(raw, "advanced", "score", "score", "NPL"),
    ):
        value = _to_int(_for_bureau(items, bureau))  # or whatever conversion _for_bureau expects
        if value is not None:
            return max(0, value)

    facilities = _find_facilities(raw)
    if facilities is None:
        return None

    wanted = {_norm_status(s) for s in cfg.non_performing_statuses}
    count = 0
    for facility in facilities:
        if not isinstance(facility, dict):
            continue
        if _norm_status(facility.get("performanceStatus")) not in wanted:
            continue
        if not cfg.count_closed_facilities and _norm_status(facility.get("status")) == "CLOSED":
            continue
        count += 1
    return count


def _extract_report_date(raw: dict[str, Any]) -> date | None:
    for key in ("report_date", "snapshot_date", "as_of", "date"):
        value = raw.get(key)
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value.strip()[:10])
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------

def create_bureau_snapshot(
    subscriber_id: str,
    lookup: SubscriberLookup,
    report_service: ReportService,
    store: SnapshotStore,
    cfg: Step2Config | None = None,
    snapshot_date: date | None = None,
) -> BureauSnapshot:
    cfg = cfg or Step2Config()
    pull_date = snapshot_date or datetime.now(WAT).date()

    bvn = lookup.get_bvn(subscriber_id)
    if not bvn or not str(bvn).strip():
        raise SnapshotError("BVN_NOT_FOUND", retryable=False)

    try:
        raw = report_service.pull(str(bvn).strip())
    except Exception as exc:
        # Stable reason only: the original message might contain personal data.
        raise SnapshotError(f"REPORT_SERVICE_ERROR:{type(exc).__name__}", retryable=True) from None
    if not raw:
        raise SnapshotError("EMPTY_REPORT", retryable=True)

    bureau = _extract_bureau(raw, cfg.default_bureau)
    score = _extract_score(raw, bureau)
    npl = _extract_non_performing(raw, cfg, bureau)
    report_date = _extract_report_date(raw)

    flags: list[str] = []
    if score is None:
        flags.append("NO_SCORE")
    if npl is None:
        flags.append("NPL_UNKNOWN")
    if report_date is not None and (pull_date - report_date).days > cfg.max_report_age_days:
        flags.append("STALE_REPORT")

    snapshot = BureauSnapshot(
        subscriber_id=subscriber_id,
        bureau=bureau,
        snapshot_date=pull_date,
        score=score,
        non_performing_count=npl or 0,
        flags=flags,
    )
    store.save(snapshot)
    return snapshot


# ---------------------------------------------------------------------------
# Storage: in-memory (tests, demos) and Mongo
# ---------------------------------------------------------------------------

class InMemorySnapshotStore:
    """Implements both SnapshotStore and SnapshotReader."""

    def __init__(self):
        self._by_key: dict[tuple[str, str, date], BureauSnapshot] = {}

    def save(self, snapshot: BureauSnapshot) -> None:
        # Same key as the Mongo unique index, so a rerun of the same day replaces, not duplicates.
        self._by_key[(snapshot.subscriber_id, snapshot.bureau, snapshot.snapshot_date)] = snapshot

    def load_previous(self, subscriber_id: str, bureau: str, before: date) -> BureauSnapshot | None:
        older = [
            s for (sid, b, d), s in self._by_key.items()
            if sid == subscriber_id and b == bureau and d < before
        ]
        return max(older, key=lambda s: s.snapshot_date) if older else None

    @property
    def saved(self) -> list[BureauSnapshot]:
        return list(self._by_key.values())


def _day_start(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


class MongoSnapshotStore:
    """Collection `bureau_snapshots`. Create this unique index once:
    collection.create_index([("subscriber_id", 1), ("bureau", 1), ("snapshot_date", 1)], unique=True)
    Old documents saved with lowercase bureau names must be migrated to the canonical names."""

    def __init__(self, collection):
        self._c = collection

    def save(self, snapshot: BureauSnapshot) -> None:
        doc = snapshot.model_dump()
        doc["snapshot_date"] = _day_start(snapshot.snapshot_date)
        doc["pulled_at"] = datetime.now(timezone.utc)
        doc["snapshot_id"] = uuid4().hex
        self._c.replace_one(
            {
                "subscriber_id": snapshot.subscriber_id,
                "bureau": snapshot.bureau,
                "snapshot_date": doc["snapshot_date"],
            },
            doc,
            upsert=True,
        )

    def load_previous(self, subscriber_id: str, bureau: str, before: date) -> BureauSnapshot | None:
        doc = self._c.find_one(
            {
                "subscriber_id": subscriber_id,
                "bureau": bureau,
                "snapshot_date": {"$lt": _day_start(before)},
            },
            {"_id": 0},
            sort=[("snapshot_date", -1)],
        )
        if not doc:
            return None
        saved_on = doc["snapshot_date"]
        return BureauSnapshot(
            subscriber_id=doc["subscriber_id"],
            bureau=doc["bureau"],
            snapshot_date=saved_on.date() if isinstance(saved_on, datetime) else saved_on,
            score=doc.get("score"),
            non_performing_count=doc.get("non_performing_count", 0),
            flags=doc.get("flags", []),
        )


def make_mongo_store(collection) -> MongoSnapshotStore:
    return MongoSnapshotStore(collection)


def make_mongo_lookup(collection) -> SubscriberLookup:
    """subscriber_id -> BVN, read from the master subscriber collection."""

    class _Lookup:
        def get_bvn(self, subscriber_id: str) -> str | None:
            doc = collection.find_one({"subscriber_id": subscriber_id}, {"bvn": 1, "_id": 0})
            return doc.get("bvn") if doc else None

    return _Lookup()


class StubReportService:
    """Placeholder until the real report-service client is wired in."""

    def pull(self, bvn: str) -> dict[str, Any]:
        raise NotImplementedError(
            "Wire the existing bureau report service here. It must accept a BVN and return a dict "
            "with score, non_performing_count (or the facility list), bureau and report_date."
        )


class StoredReportService:
    """DEV ONLY. Returns the latest report already stored for a BVN (for example the documents
    Engineer 2 saved), so you can run the whole pipeline before the real report service is wired.
    It does NOT pull anything fresh. Production replaces it with a client for the report service.
    It strips the name, and it exposes the report's own last-updated time as `report_date` so an
    old report is flagged STALE_REPORT instead of looking fresh."""

    def __init__(self, collection):
        self._c = collection

    def pull(self, bvn: str) -> dict[str, Any]:
        doc = self._c.find_one(
            {"bvn": bvn}, {"_id": 0, "name": 0, "full_name": 0}, sort=[("pulled_at", -1)]
        )
        if not doc:
            raise LookupError("NO_STORED_REPORT")
        stamp = next(
            (s for s in (
                _dig(doc, "fico", "lastUpdatedAt"),
                _dig(doc, "classic", "lastUpdatedAt"),
                _dig(doc, "firstCentral", "lastUpdatedAt"),
            ) if s is not None),
            None,
        )
        if stamp is not None:
            doc["report_date"] = stamp
        doc.pop("bvn", None)
        return doc