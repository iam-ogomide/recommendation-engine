"""
step3_pipeline.py  -  Step 3 of the CreditCliq runner (full version)

What it does, in simple words:
  For each subscriber due today, it reads their ReboundCliq repayment
  schedule and works out three facts:
    1. how many instalments were paid LATE in the last 90 days
    2. how many instalments were MISSED in the last 90 days
    3. how many facilities (credits) are NON-PERFORMING
  It returns facts only. Step 4 decides what the facts mean.

The flow:
  A. LOAD       get one subscriber's schedule rows (read-only)
  B. TRANSLATE  real field names -> simple names, UTC -> Lagos dates
  C. WINDOW     keep only instalments due in the last 90 days
  D. LABEL      on_time / late / missed / not_due
  E. COUNT      late, missed, non-performing
  F. HAND OFF   one dictionary of facts per subscriber

By default it uses FAKE data (a stub), so it is safe to run anywhere.
It only touches the real database if you set USE_REAL_DB=1.

=========================================================
DEFINITIONS  (quick reference, so we don't forget)
=========================================================
Instalment   One row in rebound_repayment_schedules: one amount due on
             one date (month_number 1, 2, 3...).
Facility     One credit / loan. All instalments of one credit share the
             same credit_ref. A subscriber can have several at once.
Due date     The day the instalment must be paid. We compare Lagos
             DATES, not exact times.
Paid         status is "PAID" AND there is a paid_at date.
Statuses     Seen so far: "PAID" and "PENDING".

Labels (for ONE instalment):
  on_time  = paid on or before the due date
  late     = paid, but AFTER the due date
  missed   = NOT paid, and the due date has passed
  not_due  = NOT paid, but the due date is still ahead

Days past due:
  late    -> paid date minus due date
  missed  -> today minus due date (it keeps growing until it is paid)

Non-performing (NPL):
  An instalment that is still unpaid MORE than 90 days after its due
  date (day 91 onwards). A facility is non-performing if ANY of its instalments is.
  90 days is the usual industry convention. It is NOT limited to the
  90-day window, because an old unpaid instalment is the most serious.

Lookback window (90 days):
  Used for late and missed counts only. Counts instalments whose
  DUE DATE falls between (today - 90 days) and today.

Money: stored in kobo. 100 kobo = 1 naira.

STILL TO CONFIRM WITH THE TEAM:
  - Is there a grace period before "late" or "missed"?
  - How are partial payments treated? (now: not paid until PAID)
  - Is 90 days the right NPL line?
"""

import os
from datetime import date, datetime, timedelta, timezone
from typing import Protocol

from core import RepaymentSummary

# =========================================================
# SETTINGS
# In the real runner these come from configuration, so the
# Product Manager can change them without a new release.
# =========================================================
LOOKBACK_DAYS = 90      # window for late and missed counts
NPL_DAYS = 90           # unpaid MORE than this many days = non-performing
SCHEDULE_COLLECTION = "rebound_repayment_schedules"
MAX_ROWS = 200          # never pull more than this per user
MAX_TIME_MS = 10000     # stop any query over 10 seconds


# =========================================================
# A. LOAD
# =========================================================
_db = None  # we connect once and reuse the connection


def _get_db():
    """Connect to MongoDB the first time we need it.

    Prefers the Rebound-specific credentials when present:
        REBOUND_MONGO_URI  +  REBOUND_MONGO_DB
    Falls back to the main runner credentials:
        MONGO_URI  +  MONGO_DB
    Credentials are never hard-coded.
    """
    global _db
    if _db is None:
        from pymongo import MongoClient
        uri = os.environ.get("REBOUND_MONGO_URI") or os.environ.get("MONGO_URI")
        db_name = os.environ.get("REBOUND_MONGO_DB") or os.environ.get("MONGO_DB")
        if not uri or not db_name:
            raise RuntimeError(
                "Set REBOUND_MONGO_URI + REBOUND_MONGO_DB "
                "(or fall back to MONGO_URI + MONGO_DB) to use the real database."
            )
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _db = client[db_name]
    return _db

def load_schedule_rows_real(user_ref, run_date, lookback_days=LOOKBACK_DAYS):
    """REAL loader. READ-ONLY: find() only, one user, a row limit, a time limit.
    We load ALL of this user's instalments (not just the window), because
    the non-performing count needs old unpaid instalments too.
    Python does the exact filtering later."""
    from bson import ObjectId

    cursor = (
        _get_db()[SCHEDULE_COLLECTION]
        .find(
            {"user_ref": ObjectId(user_ref)},
            {"credit_ref": 1, "month_number": 1, "due_date": 1,
             "total_due_kobo": 1, "amount_paid_kobo": 1,
             "status": 1, "paid_at": 1},
        )
        .limit(MAX_ROWS)
        .max_time_ms(MAX_TIME_MS)
    )
    return list(cursor)


# Fake rows for testing (same shape as the real collection)
FAKE_DATA = {}


def load_schedule_rows_stub(user_ref, run_date, lookback_days=LOOKBACK_DAYS):
    """STUB loader: returns fake rows, never touches a database."""
    return FAKE_DATA.get(user_ref, [])


def load_schedule_rows(user_ref, run_date, lookback_days=LOOKBACK_DAYS):
    """Picks the real or fake loader. Fake is the default (safe)."""
    if os.environ.get("USE_REAL_DB") == "1":
        return load_schedule_rows_real(user_ref, run_date, lookback_days)
    return load_schedule_rows_stub(user_ref, run_date, lookback_days)


# =========================================================
# B. TRANSLATE
# Only this part knows the real field names in the database.
# =========================================================
def to_lagos_date(timestamp):
    """MongoDB stores UTC. Lagos is UTC+1 all year. We compare DATES, not
    exact times, so paying 20 minutes after the due time is not 'late'.
    Anything that is not a date or datetime becomes None (bad data) instead of crashing."""
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is not None:  # make it a plain UTC time first
            timestamp = timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        return (timestamp + timedelta(hours=1)).date()
    if isinstance(timestamp, date):
        return timestamp
    return None


def clean_schedule_row(row):
    return {
        "credit_ref": str(row.get("credit_ref")),
        "month_number": row.get("month_number"),
        "due_date": to_lagos_date(row.get("due_date")),
        "paid_date": to_lagos_date(row.get("paid_at")),
        "status": row.get("status"),
    }


# =========================================================
# D. LABEL one instalment (see DEFINITIONS at the top)
# =========================================================
def label_instalment(item, run_date):
    due = item["due_date"]

    # ASSUMPTION: status "PAID" with a paid date means it was paid.
    # Any other status counts as 'not paid'.
    was_paid = str(item["status"]).strip().upper() == "PAID" and item["paid_date"] is not None

    if was_paid:
        if item["paid_date"] <= due:
            return "on_time", 0
        return "late", (item["paid_date"] - due).days

    if due < run_date:
        return "missed", (run_date - due).days
    return "not_due", 0


# =========================================================
# C + E. WINDOW and COUNT late / missed
# =========================================================
def count_late_and_missed(schedule_rows, run_date, lookback_days=LOOKBACK_DAYS):
    window_start = run_date - timedelta(days=lookback_days)

    late_count = 0
    missed_count = 0
    details = []

    for row in schedule_rows:
        item = clean_schedule_row(row)

        # Skip anything with no due date, or due outside the window
        if item["due_date"] is None:
            continue
        if item["due_date"] < window_start or item["due_date"] > run_date:
            continue

        label, days_past_due = label_instalment(item, run_date)

        if label == "late":
            late_count += 1
        elif label == "missed":
            missed_count += 1
        else:
            continue  # on_time and not_due are not counted

        details.append({
            "credit_ref": item["credit_ref"],
            "month_number": item["month_number"],
            "due_date": item["due_date"].isoformat(),
            "status": label,
            "days_past_due": days_past_due,
        })

    return {
        "late_count": late_count,
        "missed_count": missed_count,
        "details": details,
        "window": {
            "days": lookback_days,
            "from": window_start.isoformat(),
            "to": run_date.isoformat(),
        },
    }


# =========================================================
# E. COUNT non-performing facilities
# No 90-day window here: we look at ALL instalments, because an
# old unpaid one is exactly what makes a facility non-performing.
# =========================================================
def count_non_performing(schedule_rows, run_date, npl_days=NPL_DAYS):
    bad_credits = set()      # distinct facilities (credit_ref)
    bad_instalments = 0

    for row in schedule_rows:
        item = clean_schedule_row(row)
        if item["due_date"] is None:
            continue

        label, days_past_due = label_instalment(item, run_date)

        # Unpaid and overdue by MORE than npl_days (day 91 onwards)
        if label == "missed" and days_past_due > npl_days:
            bad_credits.add(item["credit_ref"])
            bad_instalments += 1

    return {
        "count": len(bad_credits),            # facilities
        "instalment_count": bad_instalments,  # instalments
        "days_rule": npl_days,
        "credit_refs": sorted(bad_credits),
    }


# =========================================================
# F. ONE subscriber: the facts step 4 will receive
# =========================================================
def run_step3(user_ref, schedule_rows, run_date):
    flags = []
    if len(schedule_rows) == 0:
        flags.append("NO_SCHEDULE_FOUND")

    return {
        "user_ref": str(user_ref),
        "snapshot_date": run_date.isoformat(),   # the date these facts are true for
        "repayments": count_late_and_missed(schedule_rows, run_date),
        "non_performing_facilities": count_non_performing(schedule_rows, run_date),
        "flags": flags,
    }


# =========================================================
# SHORT SUMMARY: just the numbers, for quick reading.
# This does NOT replace the full facts above. Step 4 still
# reads the full dictionary from run_step3.
# =========================================================
def short_summary(facts):
    return {
        "user_ref": facts["user_ref"],
        "late": facts["repayments"]["late_count"],
        "missed": facts["repayments"]["missed_count"],
        "npl_facilities": facts["non_performing_facilities"]["count"],
        "flags": facts["flags"],
    }


# =========================================================
# THE DAILY LOOP: many subscribers, one failure never stops the rest
# =========================================================
def run_daily(user_refs, run_date, loader=load_schedule_rows):
    results = []
    skipped = []

    for user_ref in user_refs:
        try:
            rows = loader(user_ref, run_date)
            results.append(run_step3(user_ref, rows, run_date))
        except Exception as error:
            # Log the KIND of error only, not the message, because
            # messages can contain personal data.
            skipped.append({"user_ref": str(user_ref),
                            "reason": type(error).__name__})

    return {"run_date": run_date.isoformat(),
            "checked": len(user_refs),
            "results": results,
            "skipped": skipped}


# =========================================================
# GLUE FOR THE MERGE: returns core.RepaymentSummary
# =========================================================
def build_repayment_summary(subscriber_id, schedule_rows, run_date,
                            lookback_days=LOOKBACK_DAYS, npl_days=NPL_DAYS):
    """Turns raw schedule rows into the RepaymentSummary that core.build_context expects.
    Rows with a missing or unreadable due date are skipped and flagged, never fatal."""
    usable, bad = [], 0
    for row in schedule_rows:
        if isinstance(row.get("due_date"), (date, datetime)):
            usable.append(row)
        else:
            bad += 1

    counts = count_late_and_missed(usable, run_date, lookback_days)
    npl = count_non_performing(usable, run_date, npl_days)
    window_start = run_date - timedelta(days=lookback_days)
    rows_in_window = sum(
        1 for r in usable
        if window_start <= clean_schedule_row(r)["due_date"] <= run_date
    )

    flags = []
    if not schedule_rows:
        flags.append("NO_SCHEDULE_FOUND")      # informational: nothing to count
    if bad:
        flags.append("SOME_ROWS_SKIPPED_BAD_DATA")

    return RepaymentSummary(
        subscriber_id=str(subscriber_id),
        window_start=window_start,
        window_end=run_date,
        rows_in_window=rows_in_window,
        late_count=counts["late_count"],
        missed_count=counts["missed_count"],
        repayment_npl_count=npl["count"],      # facilities unpaid more than 90 days past due
        flags=flags,
    )


class RepaymentSource(Protocol):
    def load(self, subscriber_id, run_date) -> list:
        """Schedule rows for one subscriber. MUST raise if the fetch fails."""
        ...


def build_repayment_summary_safe(subscriber_id, source, run_date):
    """A failed fetch must never look like a clean record: it becomes REPAYMENTS_UNAVAILABLE,
    which core.build_context refuses to evaluate."""
    try:
        rows = source.load(subscriber_id, run_date)
        if rows is None:
            raise ValueError("no rows object returned")
    except Exception:
        return RepaymentSummary(
            subscriber_id=str(subscriber_id),
            window_start=run_date - timedelta(days=LOOKBACK_DAYS),
            window_end=run_date,
            flags=["REPAYMENTS_UNAVAILABLE"],
        )
    return build_repayment_summary(subscriber_id, rows, run_date)


class StubRepaymentSource:
    """Fake source for tests and demos: {subscriber_id: [rows]}. Unknown id = no loans."""

    def __init__(self, data=None):
        self.data = data or {}

    def load(self, subscriber_id, run_date):
        return list(self.data.get(subscriber_id, []))


class MongoRepaymentSource:
    """Real source. `resolve_user_ref(subscriber_id)` maps the internal subscriber id to the
    ReboundCliq user_ref (still to be agreed). Unresolved or failing lookups raise, which the
    safe builder turns into REPAYMENTS_UNAVAILABLE."""

    def __init__(self, collection, resolve_user_ref):
        self._collection = collection
        self._resolve = resolve_user_ref

    def load(self, subscriber_id, run_date):
        user_ref = self._resolve(subscriber_id)
        if not user_ref:
            raise LookupError("USER_REF_NOT_FOUND")
        from bson import ObjectId
        cursor = (
            self._collection
            .find(
                {"user_ref": ObjectId(str(user_ref))},
                {"credit_ref": 1, "month_number": 1, "due_date": 1, "total_due_kobo": 1,
                 "amount_paid_kobo": 1, "status": 1, "paid_at": 1},
            )
            .limit(MAX_ROWS)
            .max_time_ms(MAX_TIME_MS)
        )
        return list(cursor)


def make_user_ref_resolver(collection, field: str = "user_ref"):
    """subscriber_id -> ReboundCliq user_ref. HOW these two ids link is still to be agreed with the
    team; this reads a field on the subscriber document. Returns None when it is missing, which
    makes the runner skip the subscriber with REPAYMENTS_UNAVAILABLE (never 'zero repayments')."""

    def resolve(subscriber_id: str) -> str | None:
        doc = collection.find_one({"subscriber_id": subscriber_id}, {field: 1, "_id": 0})
        value = doc.get(field) if doc else None
        return str(value) if value else None

    return resolve


# =========================================================
# TRY IT: fake subscribers (runs on fake data by default)
# =========================================================
if __name__ == "__main__":
    import json

    today = date(2026, 10, 20)

    def row(month, due, paid, status, credit="C-1"):
        return {"credit_ref": credit, "month_number": month,
                "due_date": due, "paid_at": paid, "status": status}

    FAKE_DATA["USER-A"] = [   # one late, one missed (only 8 days, not NPL)
        row(1, datetime(2026, 8, 10, 12), datetime(2026, 8, 9, 9), "PAID"),
        row(2, datetime(2026, 9, 10, 12), datetime(2026, 9, 19, 12), "PAID"),
        row(3, datetime(2026, 10, 12, 12), None, "PENDING"),
        row(4, datetime(2026, 11, 12, 12), None, "PENDING"),   # not due yet
    ]
    FAKE_DATA["USER-B"] = [   # always on time
        row(1, datetime(2026, 9, 10, 12), datetime(2026, 9, 10, 8), "PAID"),
    ]
    # USER-C has no rows at all
    FAKE_DATA["USER-D"] = [   # broken data, should be skipped, not crash
        row(1, "not-a-date", None, "PENDING"),
    ]
    FAKE_DATA["USER-E"] = [   # unpaid for 122 days: non-performing
        row(1, datetime(2026, 6, 20, 12), None, "PENDING", credit="C-9"),
    ]

    output = run_daily(["USER-A", "USER-B", "USER-C", "USER-D", "USER-E"], today)
    print(json.dumps(output, indent=2))