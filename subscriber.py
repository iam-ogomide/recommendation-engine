"""
Step 1 of the CreditCliq agentic runner
=======================================
List every subscriber who should be evaluated today, and say why anyone due
today was left out.

Rules (PRD):
  - Eligible plan
  - Subscription live (not lapsed)
  - Cycle date is today; in a short month, run on the last day of the month
    (cycle_day 31 runs on 30 Nov, and cycle_day 29-31 run on 28 Feb)

Design choices:
  - cycle_day decides when someone is due. next_run_date is only used when a
    row has no cycle_day, because next_run_date goes stale if nothing advances it.
  - Plan and status comparisons ignore case ("PRO" == "pro").
  - Only subscribers who are DUE today but left out get a skip reason. Subscribers
    who are not due today are simply not checked, so they get no decision-log row.
  - BVN is never read here.
"""
from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator

# Skip reasons (go into the decision log)
PLAN_NOT_ELIGIBLE = "PLAN_NOT_ELIGIBLE"
LAPSED = "LAPSED"
PAYMENT_NOT_OK = "PAYMENT_NOT_OK"
INVALID_ROW = "INVALID_ROW"
NO_CYCLE_DATE = "NO_CYCLE_DATE"


def _norm(value: Any) -> str:
    return str(value).strip().upper()


class Step1Config(BaseModel):
    """Starting values. Load from real config in production."""

    eligible_plans: list[str] = Field(default_factory=lambda: ["BASIC", "PRO", "PLUS"])
    active_statuses: list[str] = Field(default_factory=lambda: ["ACTIVE", "TRIALING"])
    # None = do not check payment_status. Set e.g. ["PAID"] once the PM defines "lapsed".
    ok_payment_statuses: list[str] | None = None


class SubscriberRow(BaseModel):
    """Only the fields Step 1 uses."""

    subscriber_id: str
    plan: str
    subscription_status: str
    payment_status: str | None = None
    next_run_date: date | None = None
    cycle_day: int | None = Field(default=None, ge=1, le=31)

    @field_validator("next_run_date", mode="before")
    @classmethod
    def _datetime_to_date(cls, value):
        # Mongo returns datetimes; accept any time of day.
        return value.date() if isinstance(value, datetime) else value


class SkippedSubscriber(BaseModel):
    subscriber_id: str | None
    reason: str


class Step1Result(BaseModel):
    due: list[str]
    skipped: list[SkippedSubscriber]


class SubscriberSource(Protocol):
    def find_candidates(self) -> list[dict[str, Any]]: ...


def _is_due_today(row: SubscriberRow, run_date: date) -> bool | None:
    """True/False, or None when the row has no cycle information at all."""
    if row.cycle_day is not None:
        last = calendar.monthrange(run_date.year, run_date.month)[1]
        return run_date.day == min(row.cycle_day, last)  # 31 -> 30 on a 30-day month
    if row.next_run_date is not None:
        return row.next_run_date == run_date
    return None


def select_subscribers(
    run_date: date, source: SubscriberSource, cfg: Step1Config | None = None
) -> Step1Result:
    cfg = cfg or Step1Config()
    plans = {_norm(p) for p in cfg.eligible_plans}
    statuses = {_norm(s) for s in cfg.active_statuses}
    ok_payments = (
        {_norm(p) for p in cfg.ok_payment_statuses}
        if cfg.ok_payment_statuses is not None
        else None
    )

    due: list[str] = []
    skipped: list[SkippedSubscriber] = []

    for raw in source.find_candidates():
        try:
            row = SubscriberRow.model_validate(raw)
        except Exception:
            sid = raw.get("subscriber_id") if isinstance(raw, dict) else None
            skipped.append(SkippedSubscriber(subscriber_id=sid, reason=INVALID_ROW))
            continue

        is_due = _is_due_today(row, run_date)
        if is_due is None:
            skipped.append(SkippedSubscriber(subscriber_id=row.subscriber_id, reason=NO_CYCLE_DATE))
            continue
        if not is_due:
            continue  # not checked today

        if _norm(row.plan) not in plans:
            reason = PLAN_NOT_ELIGIBLE
        elif _norm(row.subscription_status) not in statuses:
            reason = LAPSED
        elif ok_payments is not None and _norm(row.payment_status) not in ok_payments:
            reason = PAYMENT_NOT_OK
        else:
            due.append(row.subscriber_id)
            continue
        skipped.append(SkippedSubscriber(subscriber_id=row.subscriber_id, reason=reason))

    return Step1Result(due=due, skipped=skipped)


def select_due_subscribers(
    run_date: date, source: SubscriberSource, cfg: Step1Config | None = None
) -> list[str]:
    """Ids only (kept for older callers). Prefer select_subscribers, which also gives skip reasons."""
    return select_subscribers(run_date, source, cfg).due


def make_mongo_source(collection) -> SubscriberSource:
    """Wrap a pymongo collection. Reads only the fields Step 1 needs (no names, no BVN)."""

    class _MongoSource:
        def find_candidates(self) -> list[dict[str, Any]]:
            projection = {
                "subscriber_id": 1,
                "plan": 1,
                "subscription_status": 1,
                "payment_status": 1,
                "next_run_date": 1,
                "cycle_day": 1,
                "_id": 0,
            }
            return list(collection.find({}, projection))

    return _MongoSource()


if __name__ == "__main__":
    class FakeSource:
        def find_candidates(self):
            return [
                {"subscriber_id": "SUB-1", "plan": "PRO", "subscription_status": "active", "cycle_day": 23},
                {"subscriber_id": "SUB-2", "plan": "basic", "subscription_status": "ACTIVE", "cycle_day": 31},
                {"subscriber_id": "SUB-3", "plan": "FREE", "subscription_status": "active", "cycle_day": 23},
                {"subscriber_id": "SUB-4", "plan": "PRO", "subscription_status": "cancelled", "cycle_day": 23},
            ]

    for d in (date(2026, 10, 23), date(2026, 11, 30)):
        print(d, select_subscribers(d, FakeSource()).model_dump())