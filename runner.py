"""
CreditCliq Agentic Recommendation Runner: the daily orchestrator
================================================================
This file does no thinking of its own. It calls each step in order for each subscriber
and records what happened.

  Step 1   subscriber.select_subscribers        who is due today (and who was left out, with a reason)
  Step 2   credit.create_bureau_snapshot        pull the bureau report, save a dated snapshot
  Step 3   rebound.build_repayment_summary_safe + core.build_context   facts about the subscriber
  Step 4   core.evaluate_triggers               which rules fire
  Step 5   core.apply_limits                    which may be sent (cap, dismissals, repeats)
  Step 6-7 core.generate_message                Gemini writes, the validator checks, template fallback
  Step 8   RecommendationStore.save             save the record (draft or published)
  Step 10  DecisionLog.write                    one row for EVERY subscriber checked

Safety rules built in:
  - One subscriber failing never stops the run (it becomes an ERROR row).
  - A failed fetch is never treated as "zero": it skips the subscriber with a reason.
  - A missing previous snapshot because of an error is never treated as a first cycle.
  - Rerunning the same day creates no duplicates (the limits read what is already saved).
  - dry_run does everything except save recommendations.

Run the fake end-to-end demo:   python runner.py --demo
Run against MongoDB (dev):      python runner.py --only SUB-0001   (setup notes in the MongoDB section)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4
from pymongo import MongoClient
import  traceback
from pydantic import BaseModel

from core import (
    PROMPT_VERSION,
    WAT,
    GeminiClient,
    ContextNotBuildable,
    LimitsConfig,
    LLMClient,
    LLMResponse,
    PastRecommendation,
    TriggerCode,
    TriggerConfig,
    TriggerFacts,
    ValidatorConfig,
    apply_limits,
    build_context,
    build_template_message,
    evaluate_triggers,
    generate_message,
    next_send_time,
)
from credit import (
    InMemorySnapshotStore,
    MongoSnapshotStore,
    SnapshotError,
    Step2Config,
    StoredReportService,
    create_bureau_snapshot,
    make_mongo_lookup,
)
from rebound import (
    MongoRepaymentSource,
    RepaymentSource,
    StubRepaymentSource,
    build_repayment_summary_safe,
    make_user_ref_resolver,
)
from subscriber import Step1Config, make_mongo_source, select_subscribers
from dotenv import load_dotenv

load_dotenv()  # optional .env file; real environment variables take priority
# THE ONLY PLACE the MongoDB credentials are read. Everything else receives connected `db` objects.
# Main URI/DB is required. REPORT_* and REBOUND_* are optional; when omitted they fall back to main.
uri, db_name = os.environ.get("MONGO_URI"), os.environ.get("MONGO_DB")
report_uri = os.environ.get("REPORT_MONGO_URI") or uri
report_db_name = os.environ.get("REPORT_MONGO_DB") or db_name
rebound_uri = os.environ.get("REBOUND_MONGO_URI") or uri
rebound_db_name = os.environ.get("REBOUND_MONGO_DB") or db_name


# Records
class Outcome(str, Enum):
    SENT = "SENT"                # at least one recommendation was created
    NO_TRIGGER = "NO_TRIGGER"    # evaluated, nothing fired
    SUPPRESSED = "SUPPRESSED"    # something fired, the limits blocked all of it
    SKIPPED = "SKIPPED"          # could not be evaluated (reason says why)
    ERROR = "ERROR"              # unexpected failure, caught so the run continues


class RunnerConfig(BaseModel):
    auto_publish: bool = False           # False = every message is a draft for Admin review
    dry_run: bool = False                # True = do everything except save recommendations
    max_tokens_per_run: int | None = None  # cost guard: once reached, templates only


class RecommendationRecord(BaseModel):
    """The PRD record layout, plus three fields the PRD does not list yet
    (written_by, review_flag, deliver_after). Confirm those with Backend."""

    recommendation_id: str
    subscriber_id: str
    source: str = "agent"
    run_id: str
    trigger_code: str
    trigger_facts: dict[str, int | str]
    title: str
    body: str
    actions: list[dict[str, str]]
    status: str                   # "draft" or "published"
    prompt_version: str
    model_version: str
    created_at: datetime
    written_by: str               # "llm" or "template"
    review_flag: bool             # True when the template fallback was used
    deliver_after: datetime       # push window: 08:00-20:00 WAT (who enforces it is still open)


class DecisionRow(BaseModel):
    """Step 10: one row per subscriber checked, with enough detail to explain the outcome."""

    run_id: str
    subscriber_id: str
    run_date: str
    outcome: Outcome
    reasons: list[str] = []
    notes: list[str] = []         # informational (first cycle, data-quality flags, cost ceiling)
    retryable: bool = False
    bureau: str | None = None
    late_or_missed_90d: int | None = None
    repayment_npl_count: int | None = None
    non_performing_facilities: int | None = None
    triggers_fired: list[str] = []
    triggers_sent: list[str] = []
    suppressed: list[dict[str, str]] = []
    used_template: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    logged_at: str = ""


class RunSummary(BaseModel):
    run_id: str
    run_date: str
    due: int
    skipped_by_step1: int
    outcomes: dict[str, int]
    reasons: dict[str, int]
    recommendations_created: int
    templates_used: int
    input_tokens: int
    output_tokens: int
    rows: list[DecisionRow]


# ===========================================================================
# Storage interfaces (the stub Manager and the decision log plug in here)
# ===========================================================================

class RecommendationStore(Protocol):
    def save(self, record: RecommendationRecord) -> None: ...

    def history(self, subscriber_id: str) -> list[PastRecommendation]:
        """Past recommendations WITH dismissal dates. Must raise if it cannot be read."""
        ...


class InMemoryRecommendationStore:
    """Stand-in for the Recommendation Manager. The analyst's FastAPI stub wraps the same idea."""

    def __init__(self):
        self.records: list[RecommendationRecord] = []
        self._dismissed: dict[str, date] = {}

    def save(self, record: RecommendationRecord) -> None:
        self.records.append(record)

    def dismiss(self, recommendation_id: str, on: date) -> None:
        self._dismissed[recommendation_id] = on

    def history(self, subscriber_id: str) -> list[PastRecommendation]:
        return [
            PastRecommendation(
                trigger_code=TriggerCode(r.trigger_code),
                created_at=r.created_at.astimezone(WAT).date(),
                facts=r.trigger_facts,
                source=r.source,
                dismissed_at=self._dismissed.get(r.recommendation_id),
            )
            for r in self.records
            if r.subscriber_id == subscriber_id
        ]


class DecisionLog(Protocol):
    def write(self, row: DecisionRow) -> None: ...


class InMemoryDecisionLog:
    def __init__(self):
        self.rows: list[DecisionRow] = []

    def write(self, row: DecisionRow) -> None:
        self.rows.append(row)


# ===========================================================================
# Fakes so the whole pipeline runs offline
# ===========================================================================

class FakeSubscriberSource:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    def find_candidates(self) -> list[dict[str, Any]]:
        return self._rows


class FakeLookup:
    def __init__(self, mapping: dict[str, str]):
        self._map = mapping

    def get_bvn(self, subscriber_id: str) -> str | None:
        return self._map.get(subscriber_id)


class FakeReportService:
    """Keyed by BVN. An unknown BVN behaves like a service failure."""

    def __init__(self, reports: dict[str, dict[str, Any]]):
        self._reports = reports

    def pull(self, bvn: str) -> dict[str, Any]:
        if bvn not in self._reports:
            raise RuntimeError("bureau timeout")
        return self._reports[bvn]


class FakeLLM:
    """Writes the template wording from the facts it is given, so numbers always match."""

    model_version = "fake-llm-v1"

    def __init__(self, input_tokens: int = 100, output_tokens: int = 80):
        self._tokens = (input_tokens, output_tokens)

    def complete(self, system: str, user: str) -> LLMResponse:
        trigger = user.split("Trigger: ")[1].split("\n")[0]
        facts = json.loads(user.split("Facts (JSON): ")[1].split("\n")[0])
        message = build_template_message(TriggerFacts(trigger_code=TriggerCode(trigger), facts=facts))
        return LLMResponse(
            text=message.model_dump_json(),
            input_tokens=self._tokens[0],
            output_tokens=self._tokens[1],
        )


# ===========================================================================
# Wiring
# ===========================================================================

@dataclass
class RunnerDeps:
    subscribers: Any                    # Step 1 source
    lookup: Any                         # Step 2: subscriber_id -> BVN
    report_service: Any                 # Step 2: BVN -> raw report
    snapshots: Any                      # Step 2/3: save() and load_previous()
    repayments: RepaymentSource         # Step 3
    recommendations: RecommendationStore
    decision_log: DecisionLog
    llm: LLMClient
    step1_cfg: Step1Config = field(default_factory=Step1Config)
    step2_cfg: Step2Config = field(default_factory=Step2Config)
    trigger_cfg: TriggerConfig = field(default_factory=TriggerConfig)
    limits_cfg: LimitsConfig = field(default_factory=LimitsConfig)
    validator_cfg: ValidatorConfig = field(default_factory=ValidatorConfig)


@dataclass
class RunState:
    tokens_used: int = 0


def wat_today() -> date:
    """'Today' means the date in Lagos, never the server's UTC date."""
    return datetime.now(WAT).date()


# ===========================================================================
# One subscriber
# ===========================================================================

def process_one_subscriber(
    subscriber_id: str,
    run_date: date,
    run_id: str,
    deps: RunnerDeps,
    cfg: RunnerConfig,
    state: RunState,
    now: datetime,
) -> DecisionRow:
    row = DecisionRow(
        run_id=run_id,
        subscriber_id=subscriber_id,
        run_date=run_date.isoformat(),
        outcome=Outcome.SKIPPED,
    )

    def skip(reason: str) -> DecisionRow:
        row.outcome = Outcome.SKIPPED
        row.reasons = [reason]
        return row

    # ---- Step 2: bureau snapshot ----
    try:
        current = create_bureau_snapshot(
            subscriber_id, deps.lookup, deps.report_service, deps.snapshots,
            deps.step2_cfg, snapshot_date=run_date,
        )
    except SnapshotError as exc:
        row.retryable = exc.retryable
        return skip(exc.reason)
    row.bureau = current.bureau
    row.non_performing_facilities = current.non_performing_count
    row.notes += [f"SNAPSHOT:{flag}" for flag in current.flags]

    # ---- previous snapshot (an error here must NOT look like a first cycle) ----
    try:
        previous = deps.snapshots.load_previous(subscriber_id, current.bureau, run_date)
    except Exception:
        return skip("PREVIOUS_SNAPSHOT_UNAVAILABLE")

    # ---- Step 3: repayments, then the merge ----
    repayments = build_repayment_summary_safe(subscriber_id, deps.repayments, run_date)
    try:
        ctx = build_context(current, previous, repayments)
    except ContextNotBuildable as exc:
        return skip(exc.reason)
    row.late_or_missed_90d = ctx.late_or_missed_90d
    row.repayment_npl_count = ctx.repayment_npl_count
    row.notes += [f"REPAYMENTS:{flag}" for flag in repayments.flags]
    if ctx.previous_score is None:
        row.notes.append("NO_COMPARABLE_PREVIOUS_SNAPSHOT")  # SCORE_DROP cannot run

    # ---- Step 4: trigger rules ----
    fired = evaluate_triggers(ctx, deps.trigger_cfg)
    row.triggers_fired = [t.trigger_code.value for t in fired]
    if not fired:
        row.outcome = Outcome.NO_TRIGGER
        row.reasons = ["NO_TRIGGER_FIRED"]
        return row

    # ---- Step 5: sending limits (needs history; if unreadable, do not guess) ----
    try:
        history = deps.recommendations.history(subscriber_id)
    except Exception:
        return skip("HISTORY_UNAVAILABLE")
    limits = apply_limits(fired, history, run_date, deps.limits_cfg)
    row.suppressed = [
        {"trigger": s.trigger_code.value, "reason": s.reason.value} for s in limits.suppressed
    ]
    if not limits.to_send:
        row.outcome = Outcome.SUPPRESSED
        row.reasons = sorted({s.reason.value for s in limits.suppressed})
        return row

    # ---- Steps 6-7 and 8: write, validate, save ----
    for facts in limits.to_send:
        over_budget = (
            cfg.max_tokens_per_run is not None and state.tokens_used >= cfg.max_tokens_per_run
        )
        if over_budget:
            message = build_template_message(facts)
            written_by, flagged = "template", True
            prompt_version, model_version = PROMPT_VERSION, "none"
            row.notes.append("COST_CEILING_REACHED")
        else:
            written = generate_message(facts, deps.llm, deps.validator_cfg)
            message, written_by, flagged = written.message, written.source, written.flagged_for_review
            prompt_version, model_version = written.prompt_version, written.model_version
            state.tokens_used += written.input_tokens + written.output_tokens
            row.input_tokens += written.input_tokens
            row.output_tokens += written.output_tokens

        record = RecommendationRecord(
            recommendation_id=uuid4().hex,
            subscriber_id=subscriber_id,
            run_id=run_id,
            trigger_code=facts.trigger_code.value,
            trigger_facts=facts.facts,
            title=message.title,
            body=message.body,
            actions=[a.model_dump() for a in message.actions],
            status="published" if cfg.auto_publish else "draft",
            prompt_version=prompt_version,
            model_version=model_version,
            created_at=now,
            written_by=written_by,
            review_flag=flagged,
            deliver_after=next_send_time(now),
        )
        if not cfg.dry_run:
            try:
                deps.recommendations.save(record)
            except Exception:
                row.outcome = Outcome.SKIPPED
                row.reasons = ["MANAGER_UNAVAILABLE"]
                return row
        row.triggers_sent.append(facts.trigger_code.value)
        row.used_template = row.used_template or flagged

    if cfg.dry_run:
        row.notes.append("DRY_RUN_NOTHING_SAVED")
    row.outcome = Outcome.SENT
    return row


# ===========================================================================
# The daily run
# ===========================================================================

def run_daily(
    run_date: date,
    deps: RunnerDeps,
    cfg: RunnerConfig | None = None,
    now: datetime | None = None,
) -> RunSummary:
    cfg = cfg or RunnerConfig()
    now = now or datetime.now(timezone.utc)
    run_id = f"run-{run_date.isoformat()}-{uuid4().hex[:8]}"
    state = RunState()
    rows: list[DecisionRow] = []

    def log(row: DecisionRow) -> None:
        row.logged_at = datetime.now(timezone.utc).isoformat()
        rows.append(row)
        deps.decision_log.write(row)

    step1 = select_subscribers(run_date, deps.subscribers, deps.step1_cfg)

    # Subscribers who were due but left out in step 1 still get a row.
    for skipped in step1.skipped:
        log(DecisionRow(
            run_id=run_id,
            subscriber_id=skipped.subscriber_id or "UNKNOWN",
            run_date=run_date.isoformat(),
            outcome=Outcome.SKIPPED,
            reasons=[skipped.reason],
        ))

    for subscriber_id in step1.due:
        try:
            row = process_one_subscriber(subscriber_id, run_date, run_id, deps, cfg, state, now)
        except Exception as exc:  # one subscriber must never stop the run
            traceback.print_exc()  
            row = DecisionRow(
                run_id=run_id,
                subscriber_id=subscriber_id,
                run_date=run_date.isoformat(),
                outcome=Outcome.ERROR,
                reasons=[f"UNEXPECTED_ERROR:{type(exc).__name__}"],  # type only, no message
            )
        log(row)

    reasons = Counter(r for row in rows for r in row.reasons)
    return RunSummary(
        run_id=run_id,
        run_date=run_date.isoformat(),
        due=len(step1.due),
        skipped_by_step1=len(step1.skipped),
        outcomes=dict(Counter(row.outcome.value for row in rows)),
        reasons=dict(reasons),
        recommendations_created=sum(len(row.triggers_sent) for row in rows if not cfg.dry_run),
        templates_used=sum(1 for row in rows if row.used_template),
        input_tokens=sum(row.input_tokens for row in rows),
        output_tokens=sum(row.output_tokens for row in rows),
        rows=rows,
    )

# ===========================================================================
# MongoDB wiring (start with DEV credentials)
# ===========================================================================
# READS (never written):
#   - subscribers          (main cluster)
#   - report source        (Step 2 – credit bureau reports; may be a separate cluster)
#   - ReboundCliq schedules (Step 3 – repayments; may be a separate cluster)
# WRITES (runner-owned, always on main cluster):
#   - runner_bureau_snapshots, runner_decision_log
#   - runner_recommendations only with --write
# Safe defaults: dry run, offline fake model, and --only to name the subscribers.
#
# Credentials: set them as environment variables, OR put the same NAME=value lines in a file
# called .env next to runner.py (needs `pip install python-dotenv`; add .env to .gitignore).
#
# Main cluster (required) – subscribers + all runner-owned collections:
#     MONGO_URI=mongodb+srv://...
#     MONGO_DB=your_dev_db
#
# Optional separate clusters (when Step 2 / Step 3 live elsewhere):
#     REPORT_MONGO_URI=...          # credit-bureau report source (Step 2). Falls back to MONGO_URI.
#     REPORT_MONGO_DB=...           # falls back to MONGO_DB
#     REBOUND_MONGO_URI=...         # ReboundCliq repayment schedules (Step 3). Falls back to MONGO_URI.
#     REBOUND_MONGO_DB=...          # falls back to MONGO_DB
#
# Collection names:
#     REPORT_SOURCE_COLLECTION=...  (required: where pulled reports are stored)
#     SUBSCRIBERS_COLLECTION=...    (default: subscribers)
#     REBOUND_COLLECTION=...        (default: rebound_repayment_schedules)
#
# Examples:
#     python runner.py --only SUB-0001,SUB-0002              dry run
#     python runner.py --only SUB-0001 --write               also saves recommendations
#     python runner.py --only SUB-0001 --llm gemini          needs GEMINI_MODEL and GEMINI_API_KEY
#     python runner.py --setup                               creates the indexes first

class CollectionNames(BaseModel):
    # inputs (read only)
    subscribers: str = "subscribers"
    report_source: str | None = None  # required for the DEV report adapter
    rebound_schedules: str = "rebound_repayment_schedules"
    user_ref_field: str = "user_ref"  # field on the subscriber doc that holds the ReboundCliq user_ref
    # outputs (owned by the runner)
    snapshots: str = "runner_bureau_snapshots"
    recommendations: str = "runner_recommendations"
    decision_log: str = "runner_decision_log"

    @classmethod
    def from_env(cls) -> "CollectionNames":
        mapping = {
            "subscribers": "SUBSCRIBERS_COLLECTION",
            "report_source": "REPORT_SOURCE_COLLECTION",
            "rebound_schedules": "REBOUND_COLLECTION",
            "user_ref_field": "USER_REF_FIELD",
            "snapshots": "RUNNER_SNAPSHOTS",
            "recommendations": "RUNNER_RECOMMENDATIONS",
            "decision_log": "RUNNER_DECISION_LOG",
        }
        return cls(**{k: os.environ[v] for k, v in mapping.items() if os.environ.get(v)})

    def check_outputs_are_separate_from_inputs(self) -> None:
        inputs = {self.subscribers, self.report_source, self.rebound_schedules}
        for name in (self.snapshots, self.recommendations, self.decision_log):
            if name in inputs:
                raise ValueError(f"Output collection {name!r} is also an input collection. "
                                 "The runner must never write into collections it reads from.")


# ---------------------------------------------------------------------------
# Mongo-backed pieces the runner needs
# ---------------------------------------------------------------------------

def _as_utc(value: datetime) -> datetime:
    # pymongo returns naive datetimes (meaning UTC) unless tz_aware=True
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class MongoRecommendationStore:
    """Dev stand-in for the Recommendation Manager's storage. In production the Manager owns
    this data; the runner then talks to the Manager's endpoints through the same two methods."""

    def __init__(self, collection):
        self._c = collection

    def save(self, record: RecommendationRecord) -> None:
        self._c.insert_one(record.model_dump())  # datetimes are stored as native Mongo dates

    def history(self, subscriber_id: str) -> list[PastRecommendation]:
        out: list[PastRecommendation] = []
        for d in self._c.find({"subscriber_id": subscriber_id}, {"_id": 0}):
            dismissed = d.get("dismissed_at")
            if isinstance(dismissed, datetime):
                dismissed = _as_utc(dismissed).astimezone(WAT).date()
            out.append(PastRecommendation(
                trigger_code=TriggerCode(d["trigger_code"]),
                created_at=_as_utc(d["created_at"]).astimezone(WAT).date(),
                facts=d["trigger_facts"],
                source=d.get("source", "agent"),
                dismissed_at=dismissed,
            ))
        return out

    def dismiss(self, recommendation_id: str, when: datetime | None = None) -> None:
        """Dev helper that simulates the subscriber tapping Dismiss in the app."""
        self._c.update_one(
            {"recommendation_id": recommendation_id},
            {"$set": {"dismissed_at": when or datetime.now(timezone.utc)}},
        )


class MongoDecisionLog:
    def __init__(self, collection):
        self._c = collection

    def write(self, row: DecisionRow) -> None:
        self._c.insert_one(row.model_dump(mode="json"))  # enums become plain strings


class _OnlySource:
    """Restrict a run to named subscribers (handy in dev)."""

    def __init__(self, inner, ids: set[str]):
        self._inner, self._ids = inner, ids

    def find_candidates(self):
        return [r for r in self._inner.find_candidates() if r.get("subscriber_id") in self._ids]


# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------

def build_mongo_deps(
    main_db,
    report_db,
    rebound_db,
    names: CollectionNames,
    llm,
    only: set[str] | None = None,
) -> RunnerDeps:
    """Wire the runner to (possibly different) Mongo databases.

    main_db      – subscribers + runner-owned collections (snapshots, recommendations, decision_log)
    report_db    – credit-bureau report source (Step 2)
    rebound_db   – ReboundCliq repayment schedules (Step 3)
    """
    names.check_outputs_are_separate_from_inputs()
    if not names.report_source:
        raise ValueError("REPORT_SOURCE_COLLECTION is not set (where are the pulled reports stored?)")

    subscribers = main_db[names.subscribers]
    source = make_mongo_source(subscribers)
    if only:
        source = _OnlySource(source, only)

    return RunnerDeps(
        subscribers=source,
        lookup=make_mongo_lookup(subscribers),
        report_service=StoredReportService(report_db[names.report_source]),
        snapshots=MongoSnapshotStore(main_db[names.snapshots]),
        repayments=MongoRepaymentSource(
            rebound_db[names.rebound_schedules],
            make_user_ref_resolver(subscribers, names.user_ref_field),
        ),
        recommendations=MongoRecommendationStore(main_db[names.recommendations]),
        decision_log=MongoDecisionLog(main_db[names.decision_log]),
        llm=llm,
    )


def setup_indexes(db, names: CollectionNames) -> None:
    """Indexes are only needed on the main (runner-owned) database."""
    db[names.snapshots].create_index(
        [("subscriber_id", 1), ("bureau", 1), ("snapshot_date", 1)], unique=True
    )
    db[names.recommendations].create_index([("subscriber_id", 1), ("created_at", -1)])
    db[names.recommendations].create_index("recommendation_id", unique=True)
    db[names.decision_log].create_index([("run_id", 1), ("subscriber_id", 1)])


def run_with_db(
    main_db,
    report_db,
    rebound_db,
    names: CollectionNames,
    run_date: date,
    *,
    llm=None,
    only: set[str] | None = None,
    write: bool = False,
    publish: bool = False,
    max_tokens: int | None = None,
    now: datetime | None = None,
) -> RunSummary:
    deps = build_mongo_deps(main_db, report_db, rebound_db, names, llm or FakeLLM(), only)
    cfg = RunnerConfig(dry_run=not write, auto_publish=publish, max_tokens_per_run=max_tokens)
    return run_daily(run_date, deps, cfg, now=now)

# ===========================================================================
# Fake end-to-end demo (no network, no Mongo, no real model)
# ===========================================================================

def run_fake_e2e() -> None:
    run_date = date(2026, 10, 23)

    subscribers = FakeSubscriberSource([
        {"subscriber_id": "SUB-A", "plan": "PRO", "subscription_status": "active", "cycle_day": 23},
        {"subscriber_id": "SUB-B", "plan": "BASIC", "subscription_status": "active", "cycle_day": 23},
        {"subscriber_id": "SUB-C", "plan": "PRO", "subscription_status": "cancelled", "cycle_day": 23},
        {"subscriber_id": "SUB-D", "plan": "FREE", "subscription_status": "active", "cycle_day": 23},
        {"subscriber_id": "SUB-E", "plan": "PRO", "subscription_status": "active", "cycle_day": 23},
        {"subscriber_id": "SUB-F", "plan": "PRO", "subscription_status": "active", "cycle_day": 5},
    ])
    lookup = FakeLookup({"SUB-A": "111", "SUB-B": "222", "SUB-E": "555"})
    report_service = FakeReportService({
        "111": {"score": 580, "non_performing_count": 0, "bureau": "FIRST_CENTRAL"},
        "222": {"score": 720, "non_performing_count": 1, "bureau": "FIRST_CENTRAL"},
        # "555" is missing on purpose: the bureau call fails for SUB-E
    })

    snapshots = InMemorySnapshotStore()
    from core import BureauSnapshot
    snapshots.save(BureauSnapshot(
        subscriber_id="SUB-A", bureau="FIRST_CENTRAL",
        snapshot_date=date(2026, 9, 23), score=650,
    ))

    def sched(month, due, paid, status):
        return {"credit_ref": "C-1", "month_number": month, "due_date": due, "paid_at": paid, "status": status}

    repayments = StubRepaymentSource({
        "SUB-B": [
            sched(1, datetime(2026, 8, 10, 12), datetime(2026, 8, 25, 12), "PAID"),   # paid late
            sched(2, datetime(2026, 9, 10, 12), None, "PENDING"),                      # missed
        ],
    })

    recommendations = InMemoryRecommendationStore()
    decision_log = InMemoryDecisionLog()
    deps = RunnerDeps(
        subscribers=subscribers, lookup=lookup, report_service=report_service,
        snapshots=snapshots, repayments=repayments,
        recommendations=recommendations, decision_log=decision_log, llm=FakeLLM(),
    )

    summary = run_daily(run_date, deps)

    print(f"{summary.run_id}: {summary.due} due, {summary.skipped_by_step1} left out in step 1")
    print("\nDecision log (one row per subscriber checked):")
    for row in summary.rows:
        print(f"  {row.subscriber_id:6} {row.outcome.value:10} reasons={row.reasons} "
              f"sent={row.triggers_sent} notes={row.notes}")
    print("\nSaved recommendations:")
    for rec in recommendations.records:
        print(f"  {rec.subscriber_id} {rec.trigger_code:15} [{rec.status}] {rec.title!r} "
              f"written_by={rec.written_by} deliver_after={rec.deliver_after:%Y-%m-%d %H:%M}")
    print("\nOutcome counts:", summary.outcomes)

    print("\nRunning the SAME day again (should create nothing new):")
    again = run_daily(run_date, deps)
    print("  outcomes:", again.outcomes, "| records now:", len(recommendations.records))


# ===========================================================================
# Command line
# ===========================================================================

def load_env() -> bool:
    """Credentials come from environment variables. If python-dotenv is installed and a .env file
    exists in this folder (or a parent), load it too. Real environment variables always win over
    the file. Keep .env out of Git (add it to .gitignore)."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return False
    path = find_dotenv(usecwd=True)
    return load_dotenv(path) if path else False


def _print_summary(summary: RunSummary, write: bool) -> None:
    print(f"\n{summary.run_id}: {summary.due} due, {summary.skipped_by_step1} left out in step 1")
    for row in summary.rows:
        print(f"  {row.subscriber_id:12} {row.outcome.value:10} reasons={row.reasons} "
              f"sent={row.triggers_sent} notes={row.notes}")
    print("\nOutcomes:", summary.outcomes)
    print(f"Recommendations saved: {summary.recommendations_created} "
          f"({'write mode' if write else 'dry run: nothing saved'}) | templates used: {summary.templates_used}")
    print(f"Tokens in/out: {summary.input_tokens}/{summary.output_tokens}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the recommendation runner against MongoDB (dev).")
    p.add_argument("--date", help="run date YYYY-MM-DD (default: today in Lagos)")
    p.add_argument("--only", help="comma-separated subscriber ids to include")
    p.add_argument("--write", action="store_true", help="save recommendations (default is a dry run)")
    p.add_argument("--publish", action="store_true", help="save as published instead of draft")
    p.add_argument("--llm", choices=["fake", "gemini"], default="fake")
    p.add_argument("--max-tokens", type=int, help="token ceiling for the run")
    p.add_argument("--demo", action="store_true", help="run the offline fake demo and exit")
    p.add_argument("--setup", action="store_true", help="create indexes on the runner-owned collections")
    args = p.parse_args(argv)

    if args.demo:
        run_fake_e2e()
        return 0


    if not uri or not db_name:
        print("Set MONGO_URI and MONGO_DB first (see the MongoDB section in runner.py).")
        return 2
    try:
        names = CollectionNames.from_env()
        names.check_outputs_are_separate_from_inputs()
    except ValueError as exc:
        print(exc)
        return 2
    if not names.report_source:
        print("Set REPORT_SOURCE_COLLECTION to the collection that holds the pulled reports.")
        return 2

    llm = FakeLLM()
    if args.llm == "gemini":
        model = os.environ.get("GEMINI_MODEL")
        if not model:
            print("Set GEMINI_MODEL (exact model id) and GEMINI_API_KEY to use --llm gemini.")
            return 2
        llm = GeminiClient(model=model)

    # ---- Connect to the (possibly different) Mongo clusters ----
    def _connect(label: str, u: str, dbn: str):
        try:
            c = MongoClient(u, serverSelectionTimeoutMS=5000, tz_aware=True)
            c.admin.command("ping")
            return c, c[dbn]
        except Exception as exc:
            print(f"Could not connect to {label} MongoDB ({type(exc).__name__}). "
                  f"Check the URI, network and IP allow-list.")
            raise SystemExit(1) from None

    main_client, main_db = _connect("main", uri, db_name)

    # Re-use the main client when the URI is the same (avoids extra connections).
    if report_uri == uri:
        report_db = main_client[report_db_name]
    else:
        _, report_db = _connect("report (Step 2)", report_uri, report_db_name)

    if rebound_uri == uri:
        rebound_db = main_client[rebound_db_name]
    else:
        _, rebound_db = _connect("rebound (Step 3)", rebound_uri, rebound_db_name)

    print(f"Connected:")
    print(f"  main     → db {db_name!r}          (subscribers + runner collections)")
    print(f"  report   → db {report_db_name!r}   (Step 2 – credit bureau reports) "
          f"{'(same cluster)' if report_uri == uri else '(separate cluster)'}")
    print(f"  rebound  → db {rebound_db_name!r}  (Step 3 – ReboundCliq repayments) "
          f"{'(same cluster)' if rebound_uri == uri else '(separate cluster)'}")
    print(f"  reads  : {names.subscribers}, {names.report_source}, {names.rebound_schedules}")
    print(f"  writes : {names.snapshots}, {names.decision_log}"
          + (f", {names.recommendations}" if args.write else f"  ({names.recommendations}: dry run, not written)"))

    if args.setup:
        setup_indexes(main_db, names)
        print("Indexes created on main database.")

    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    run_date = date.fromisoformat(args.date) if args.date else wat_today()
    summary = run_with_db(
        main_db, report_db, rebound_db, names, run_date,
        llm=llm, only=only, write=args.write,
        publish=args.publish, max_tokens=args.max_tokens,
    )
    _print_summary(summary, args.write)
    return 0


if __name__ == "__main__":
    sys.exit(main())