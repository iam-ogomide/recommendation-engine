"""Recommendation runner

Sections, in dependency order:
  MODELS, CONFIG, CONTEXT BUILDER (PRD step 3, the merge),
  TRIGGER RULES (PRD step 4), SENDING LIMITS (PRD step 5),
  PUSH WINDOW (PRD step 5, timing rule),
  VALIDATOR (PRD step 7, the check), WRITER (PRD steps 6 and 7, the flow),
  GEMINI CLIENT (PRD step 6, the real provider).
"""
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError, model_validator

# ======================================================================
# MODELS
# ======================================================================

class TriggerCode(str, Enum):
    SCORE_DROP = "SCORE_DROP"
    LOW_SCORE = "LOW_SCORE"
    POOR_REPAYMENT = "POOR_REPAYMENT"


class SubscriberContext(BaseModel):
    """Everything the rules need about one subscriber for one run."""

    subscriber_id: str
    bureau: str  # e.g. "firstcentral"
    current_score: int
    current_snapshot_date: date
    # None on a first cycle, or when the selected bureau changed
    previous_score: int | None = None
    previous_snapshot_date: date | None = None
    late_or_missed_90d: int = 0
    non_performing_facilities: int = 0
    # From ReboundCliq: repayments unpaid more than 90 days past due. NOT the bureau count above.
    repayment_npl_count: int = 0


class TriggerFacts(BaseModel):
    """What a fired rule returns. These numbers are the only facts the writer sees."""

    trigger_code: TriggerCode
    facts: dict[str, int | str]


class Action(BaseModel):
    label: str
    screen: str


class GeneratedMessage(BaseModel):
    title: str = Field(max_length=60)
    body: str = Field(max_length=400)
    actions: list[Action] = Field(max_length=3)


class PastRecommendation(BaseModel):
    """One earlier recommendation for a subscriber, as the runner reads it back.
    Dismissals live in the Manager, so dismissed_at must come from there."""

    trigger_code: TriggerCode
    created_at: date
    facts: dict[str, int | str]
    source: str = "agent"  # only "agent" items count toward the monthly cap
    dismissed_at: date | None = None


class SuppressionReason(str, Enum):
    TRIGGER_ALREADY_SENT_THIS_MONTH = "TRIGGER_ALREADY_SENT_THIS_MONTH"
    DISMISSED_QUIET_PERIOD = "DISMISSED_QUIET_PERIOD"
    IDENTICAL_FACTS = "IDENTICAL_FACTS"
    MONTHLY_CAP_REACHED = "MONTHLY_CAP_REACHED"


class Suppressed(BaseModel):
    trigger_code: TriggerCode
    reason: SuppressionReason


class LimitsResult(BaseModel):
    """Output of the sending limits. `suppressed` feeds the decision log."""

    to_send: list[TriggerFacts]
    suppressed: list[Suppressed]


class BureauSnapshot(BaseModel):
    """One stored bureau pull for one subscriber (step 2's output)."""

    subscriber_id: str
    bureau: str  # canonical names, e.g. FIRST_CENTRAL, CRC, CREDIT_REGISTRY
    snapshot_date: date
    score: int | None = None  # None when the bureau gave no usable score
    non_performing_count: int = 0  # facilities marked non-performing on this report


class RepaymentSummary(BaseModel):
    """Repayment half of step 3 (from ReboundCliq)."""

    subscriber_id: str
    window_start: date
    window_end: date
    rows_in_window: int = 0
    late_count: int = 0  # paid after the due date, due date inside the window
    missed_count: int = 0  # unpaid, due date passed, due date inside the window
    repayment_npl_count: int = 0  # unpaid more than 90 days past due (no lookback window)
    # "REPAYMENTS_UNAVAILABLE" means the fetch failed: the counts above must not be trusted.
    flags: list[str] = []


class ContextNotBuildable(Exception):
    """Raised when a subscriber cannot be evaluated today. `reason` goes in the decision log."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ======================================================================
# CONFIG
# ======================================================================

class TriggerConfig(BaseModel):
    """Starting values from the PRD. In production, load these from config
    so the PM can change them without a release."""

    score_drop_min_points: int = 1
    low_score_threshold: int = 600
    # Optional per-bureau override, e.g. {"crc": 550}. Falls back to the default.
    low_score_threshold_by_bureau: dict[str, int] = {}
    poor_repayment_min_late: int = 2
    poor_repayment_window_days: int = 90
    poor_repayment_min_non_performing: int = 1

    def low_score_line(self, bureau: str) -> int:
        return self.low_score_threshold_by_bureau.get(bureau, self.low_score_threshold)


class LimitsConfig(BaseModel):
    max_agent_messages_per_month: int = 2
    dismissal_quiet_days: int = 60
    # Highest priority first, used when more triggers fire than the cap allows
    priority: list[str] = ["POOR_REPAYMENT", "SCORE_DROP", "LOW_SCORE"]


# ======================================================================
# CONTEXT BUILDER (PRD step 3, the merge)
# ======================================================================
# Joins today's bureau snapshot, last month's snapshot and the repayment summary into
# the SubscriberContext the rules need. Pure function: no database, no network.


def build_context(
    current: BureauSnapshot,
    previous: BureauSnapshot | None,
    repayments: RepaymentSummary,
) -> SubscriberContext:
    if current.score is None:
        raise ContextNotBuildable("NO_CURRENT_SCORE")
    if "REPAYMENTS_UNAVAILABLE" in repayments.flags:
        # Never treat a failed fetch as "zero late payments".
        raise ContextNotBuildable("REPAYMENTS_UNAVAILABLE")
    if current.subscriber_id != repayments.subscriber_id or (
        previous is not None and previous.subscriber_id != current.subscriber_id
    ):
        raise ContextNotBuildable("SUBSCRIBER_MISMATCH")

    # The previous snapshot only counts if it is comparable: same bureau, a real
    # score, and an earlier date. Otherwise this is treated as a first cycle.
    usable_previous = (
        previous is not None
        and previous.bureau == current.bureau
        and previous.score is not None
        and previous.snapshot_date < current.snapshot_date
    )

    return SubscriberContext(
        subscriber_id=current.subscriber_id,
        bureau=current.bureau,
        current_score=current.score,
        current_snapshot_date=current.snapshot_date,
        previous_score=previous.score if usable_previous else None,
        previous_snapshot_date=previous.snapshot_date if usable_previous else None,
        late_or_missed_90d=repayments.late_count + repayments.missed_count,
        non_performing_facilities=current.non_performing_count,
        repayment_npl_count=repayments.repayment_npl_count,
    )


# ======================================================================
# TRIGGER RULES 
# ======================================================================

def check_score_drop(ctx: SubscriberContext, cfg: TriggerConfig) -> TriggerFacts | None:
    # First cycle, or bureau changed: no comparable previous snapshot.
    if ctx.previous_score is None or ctx.previous_snapshot_date is None:
        return None

    points_dropped = ctx.previous_score - ctx.current_score
    if points_dropped < cfg.score_drop_min_points:
        return None

    return TriggerFacts(
        trigger_code=TriggerCode.SCORE_DROP,
        facts={
            "previous_score": ctx.previous_score,
            "current_score": ctx.current_score,
            "points_dropped": points_dropped,
            "previous_snapshot_date": ctx.previous_snapshot_date.isoformat(),
            "current_snapshot_date": ctx.current_snapshot_date.isoformat(),
        },
    )


def check_low_score(ctx: SubscriberContext, cfg: TriggerConfig) -> TriggerFacts | None:
    line = cfg.low_score_line(ctx.bureau)
    if ctx.current_score >= line:
        return None

    return TriggerFacts(
        trigger_code=TriggerCode.LOW_SCORE,
        facts={"current_score": ctx.current_score, "score_line": line},
    )


def check_poor_repayment(ctx: SubscriberContext, cfg: TriggerConfig) -> TriggerFacts | None:
    too_many_late = ctx.late_or_missed_90d >= cfg.poor_repayment_min_late
    has_non_performing = (
        ctx.non_performing_facilities >= cfg.poor_repayment_min_non_performing
    )
    if not (too_many_late or has_non_performing):
        return None

    return TriggerFacts(
        trigger_code=TriggerCode.POOR_REPAYMENT,
        facts={
            "late_or_missed_count": ctx.late_or_missed_90d,
            "non_performing_count": ctx.non_performing_facilities,
            "window_days": cfg.poor_repayment_window_days,
        },
    )


def evaluate_triggers(ctx: SubscriberContext, cfg: TriggerConfig) -> list[TriggerFacts]:
    """Run every rule and return the ones that fired.
    Ordering and the monthly cap are the sending limits' job (next step)."""
    results = [
        check_score_drop(ctx, cfg),
        check_low_score(ctx, cfg),
        check_poor_repayment(ctx, cfg),
    ]
    return [r for r in results if r is not None]


# ======================================================================
# SENDING LIMITS
# ======================================================================

def _same_month(a: date, b: date) -> bool:
    return (a.year, a.month) == (b.year, b.month)


def apply_limits(
    fired: list[TriggerFacts],
    history: list[PastRecommendation],
    today: date,
    cfg: LimitsConfig,
) -> LimitsResult:
    """PRD step 5. Decide which fired triggers may be sent today.

    Assumes "month" means the calendar month (open question with the PM).
    Only agent-sourced history counts; admin posts are ignored.
    """
    agent_history = [h for h in history if h.source == "agent"]
    rank = {code: i for i, code in enumerate(cfg.priority)}
    ordered = sorted(fired, key=lambda t: rank.get(t.trigger_code.value, len(rank)))

    suppressed: list[Suppressed] = []
    eligible: list[TriggerFacts] = []

    for trig in ordered:
        past = [h for h in agent_history if h.trigger_code == trig.trigger_code]

        reason: SuppressionReason | None = None
        if any(_same_month(h.created_at, today) for h in past):
            reason = SuppressionReason.TRIGGER_ALREADY_SENT_THIS_MONTH
        elif any(
            h.dismissed_at is not None
            and (today - h.dismissed_at).days < cfg.dismissal_quiet_days
            for h in past
        ):
            reason = SuppressionReason.DISMISSED_QUIET_PERIOD
        elif any(h.facts == trig.facts for h in past):
            # Also covers "a LOW_SCORE repeat needs a changed score"
            reason = SuppressionReason.IDENTICAL_FACTS

        if reason:
            suppressed.append(Suppressed(trigger_code=trig.trigger_code, reason=reason))
        else:
            eligible.append(trig)

    already_sent = sum(1 for h in agent_history if _same_month(h.created_at, today))
    slots = max(cfg.max_agent_messages_per_month - already_sent, 0)

    to_send = eligible[:slots]
    for trig in eligible[slots:]:
        suppressed.append(
            Suppressed(
                trigger_code=trig.trigger_code,
                reason=SuppressionReason.MONTHLY_CAP_REACHED,
            )
        )

    return LimitsResult(to_send=to_send, suppressed=suppressed)


# ======================================================================
# PUSH WINDOW (PRD step 5, timing rule)
# ======================================================================
# Push messages go out between 08:00 and 20:00 West Africa Time. Anything created
# outside that window waits for the next 08:00. This decides WHEN, not WHETHER, so
# it is separate from apply_limits. Who calls it (runner or Manager) is still open
# with Backend; the runner can store the result as the record's deliver-after time.

WAT = timezone(timedelta(hours=1))  # UTC+1, no daylight saving


class PushWindowConfig(BaseModel):
    start_hour: int = Field(default=8, ge=0, le=23)
    end_hour: int = Field(default=20, ge=1, le=23)  # exclusive: 20:00 is outside

    @model_validator(mode="after")
    def _start_before_end(self):
        if self.start_hour >= self.end_hour:
            raise ValueError("start_hour must be before end_hour")
        return self


def is_within_push_window(moment: datetime, cfg: PushWindowConfig | None = None) -> bool:
    cfg = cfg or PushWindowConfig()
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware")
    local = moment.astimezone(WAT)
    return cfg.start_hour <= local.hour < cfg.end_hour


def next_send_time(created_at: datetime, cfg: PushWindowConfig | None = None) -> datetime:
    """Earliest allowed push time (in WAT) for something created at `created_at`."""
    cfg = cfg or PushWindowConfig()
    if created_at.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    local = created_at.astimezone(WAT)
    opens = local.replace(hour=cfg.start_hour, minute=0, second=0, microsecond=0)
    closes = local.replace(hour=cfg.end_hour, minute=0, second=0, microsecond=0)
    if local < opens:
        return opens
    if local < closes:
        return local
    return opens + timedelta(days=1)


# ======================================================================
# VALIDATOR
# ======================================================================
# Decides whether generated text is safe to save: JSON format and length limits,
# every number appears in the facts, banned phrases, PII patterns, allowed screens.
# Topic rules (promises, bureau method, guessed causes, legal/tax/investment advice,
# lender names) are keyword/regex checks. They catch obvious wording, not meaning, so
# a paraphrase can slip through. The prompt, the eval set and Admin review back them up.
# Not checkable by code: naming another subscriber (the writer never receives names).

# Starter lists. They catch obvious wording only (see the note in the section header),
# so the PM, compliance and the eval set should extend them over time.
_STARTER_PATTERNS: dict[str, list[str]] = {
    "promise": [
        r"\bwill (?:increase|improve|raise|boost|lift|recover|go up|rise)\b",
        r"\byou(?:'ll| will) (?:qualify|get (?:a )?loan|be (?:eligible|accepted))\b",
        r"\bapproval\b",
        r"\bwithin (?:the next |a |just a )?(?:few )?(?:days|weeks|months)\b",
    ],
    "bureau_method": [
        r"\balgorithm",
        r"\bformula\b",
        r"\bcalculat(?:e|es|ed|ion)\b",
        r"\bweight(?:s|ed|ing)?\b",
        r"\bscoring (?:model|method|criteria)\b",
        r"\bdetermined by\b",
    ],
    "speculated_cause": [
        r"\bdue to\b",
        r"\bcaused by\b",
        r"\bas a result of\b",
        r"\bloan application",
        r"\bhard (?:pull|inquiry|enquiry)\b",
        r"\b(?:inquiry|inquiries|enquiry|enquiries)\b",
    ],
    "advice": [
        r"\b(?:lawyer|attorney|solicitor|lawsuit|litigation|court)\b",
        r"\blegal (?:action|advice|rights)\b",
        r"\b(?:tax|taxes|taxation|firs|vat)\b",
        r"\b(?:invest|investing|investment|investments|stocks?|securities|crypto|bitcoin)\b",
        r"\b(?:mutual funds?|treasury bills?|bankruptcy|insolvency)\b",
    ],
}

# STARTER list of lender names to block. Replace with the real lenders in your system.
_STARTER_LENDERS = [
    "GTBank", "GT Bank", "Guaranty Trust", "Access Bank", "Zenith", "UBA",
    "United Bank for Africa", "First Bank", "FirstBank", "FCMB", "Fidelity Bank",
    "Sterling Bank", "Wema", "Union Bank", "Polaris", "Stanbic", "Ecobank",
    "Opay", "Palmpay", "Kuda", "Moniepoint", "FairMoney", "Renmoney",
]

_CATEGORY_GUIDANCE = {
    "promise": "do not promise a higher score, an approval or a timeline",
    "bureau_method": "do not describe how a bureau calculates its score",
    "speculated_cause": "do not guess why the score changed",
    "advice": "do not give legal, tax or investment advice",
    "lender": "do not name any lender",
}


class ValidatorConfig(BaseModel):
    # PRD starting list; the PM can extend it.
    banned_phrases: list[str] = ["guarantee", "will increase your score", "approved"]
    # Regex patterns by category (case-insensitive). Keys match _CATEGORY_GUIDANCE.
    banned_patterns: dict[str, list[str]] = _STARTER_PATTERNS
    lender_names: list[str] = _STARTER_LENDERS

    @model_validator(mode="after")
    def _patterns_must_compile(self):
        for category, patterns in self.banned_patterns.items():
            for p in patterns:
                try:
                    re.compile(p)
                except re.error as exc:
                    raise ValueError(f"bad regex in banned_patterns[{category!r}]: {p!r} ({exc})")
        return self

    # PLACEHOLDER screen ids. Replace with the real list from Mobile.
    allowed_screens: dict[str, list[str]] = {
        "SCORE_DROP": ["credit_report", "improve_score"],
        "LOW_SCORE": ["improve_score", "credit_report"],
        "POOR_REPAYMENT": ["repayments", "improve_score"],
    }


class ValidationResult(BaseModel):
    ok: bool
    message: GeneratedMessage | None = None
    errors: list[str] = []


_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_LONG_DIGITS = re.compile(r"\d{10,}")  # BVN (11), account no. (10), bare phone numbers
_NG_PHONE = re.compile(r"(\+?234[\s-]?|\b0)[789][01]\d[\s-]?\d{3}[\s-]?\d{4}\b")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _numbers_in(text: str) -> set[float]:
    return {float(m.replace(",", "")) for m in _NUMBER.findall(text)}


def _allowed_numbers(facts: TriggerFacts) -> set[float]:
    allowed: set[float] = set()
    for value in facts.facts.values():
        if isinstance(value, int):
            allowed.add(float(abs(value)))
        else:  # e.g. an ISO date: 2026-10-10 -> 2026, 10, 10
            allowed |= _numbers_in(str(value))
    return allowed


def validate_message(
    raw: str, facts: TriggerFacts, cfg: ValidatorConfig | None = None
) -> ValidationResult:
    cfg = cfg or ValidatorConfig()

    # 1. Format and length limits
    try:
        data = json.loads(_FENCE.sub("", raw.strip()))
    except json.JSONDecodeError:
        return ValidationResult(ok=False, errors=["format: output is not valid JSON"])
    try:
        msg = GeneratedMessage.model_validate(data)
    except ValidationError as exc:
        errs = [
            f"format: {'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        ]
        return ValidationResult(ok=False, errors=errs)

    errors: list[str] = []
    if not msg.title.strip() or not msg.body.strip():
        errors.append("format: title and body must not be empty")
    if any(not a.label.strip() or not a.screen.strip() for a in msg.actions):
        errors.append("format: every action needs a label and a screen")

    text = " ".join([msg.title, msg.body, *(a.label for a in msg.actions)])

    # 2. Every number must come from the facts
    extra = sorted(_numbers_in(text) - _allowed_numbers(facts))
    if extra:
        shown = ", ".join(f"{n:g}" for n in extra)
        errors.append(f"numbers not in the facts: {shown}")

    # 3. Banned phrases
    lowered = text.lower()
    for phrase in cfg.banned_phrases:
        if phrase.lower() in lowered:
            errors.append(f"banned phrase: {phrase}")

    # 3b. Topic patterns: promises, bureau method, guessed causes, advice, lender names
    for category, patterns in cfg.banned_patterns.items():
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                guidance = _CATEGORY_GUIDANCE.get(category, "remove this wording")
                errors.append(f"{category}: {guidance} (matched {m.group(0)!r})")
                break  # one report per category is enough
    if cfg.lender_names:
        lender_re = r"\b(?:" + "|".join(re.escape(n) for n in cfg.lender_names) + r")\b"
        m = re.search(lender_re, text, re.IGNORECASE)
        if m:
            errors.append(f"lender: {_CATEGORY_GUIDANCE['lender']} (matched {m.group(0)!r})")

    # 4. PII patterns
    if _LONG_DIGITS.search(text) or _NG_PHONE.search(text):
        errors.append("pii: text contains something that looks like a BVN, account or phone number")

    # 5. Action screens must be on the allowed list
    allowed = cfg.allowed_screens.get(facts.trigger_code.value)
    if allowed is not None:
        for a in msg.actions:
            if a.screen not in allowed:
                errors.append(f"action screen not allowed: {a.screen}")

    if errors:
        return ValidationResult(ok=False, errors=errors)
    return ValidationResult(ok=True, message=msg)


# ======================================================================
# WRITER
# ======================================================================
# One LLM call, validate, retry once, then fall back to the PM's template wording.
# The writer only receives the trigger facts, never a subscriber id, name, BVN,
# phone or account number. Any token swap belongs inside the LLMClient implementation.

PROMPT_VERSION = "v1"
MAX_ATTEMPTS = 2  # first try plus one retry


# ---------- LLM interface (provider-agnostic) ----------

class LLMResponse(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    model_version: str  # pin an exact version, not a floating alias

    def complete(self, system: str, user: str) -> LLMResponse: ...


# ---------- Prompts ----------

_SYSTEM = """You write short in-app recommendations for CreditCliq subscribers about their credit report.

Reply with ONLY a JSON object, no markdown and no code fences:
{{"title": string (max 60 characters), "body": string (max 400 characters), "actions": [{{"label": string, "screen": string}}] (0 to 3 items)}}

Rules:
- Open the body with the reason the subscriber is hearing from us, in plain words.
- Use only numbers that appear in the facts, written as digits. Never compute or invent a number.
- Never state or work out a score other than the ones in the facts.
- Never promise a higher score, an approval or a timeline. Never use the words "guarantee" or "approved".
- Never explain how a bureau calculates its score, and never guess why a score changed.
- Never mention other people or lenders, and never give legal, tax or investment advice.
- Never include a BVN, account number or phone number.
- Each action's screen must be one of: {screens}.
Tone: warm, clear and short."""

_GUIDANCE = {
    TriggerCode.SCORE_DROP: (
        "Say the score dropped. Give the previous and current score and both dates. "
        "Do not guess a cause."
    ),
    TriggerCode.LOW_SCORE: (
        "Say the current score is below the score line in the facts. "
        "Encourage practical next steps without promising any result."
    ),
    TriggerCode.POOR_REPAYMENT: (
        "Say how many late or missed repayments and how many non-performing facilities "
        "were found, and over what window. Encourage catching up on repayments."
    ),
}


def build_system_prompt(facts: TriggerFacts, vcfg: ValidatorConfig) -> str:
    screens = vcfg.allowed_screens.get(facts.trigger_code.value, [])
    return _SYSTEM.format(screens=", ".join(screens) or "none")


def build_user_prompt(facts: TriggerFacts, previous_errors: list[str] | None = None) -> str:
    parts = [
        f"Trigger: {facts.trigger_code.value}",
        f"Facts (JSON): {json.dumps(facts.facts)}",
        _GUIDANCE[facts.trigger_code],
    ]
    if previous_errors:
        parts.append(
            "Your previous answer was rejected for these reasons. Fix them and answer again:\n- "
            + "\n- ".join(previous_errors)
        )
    return "\n".join(parts)


# ---------- Template fallback ----------
# PLACEHOLDER wording. The Product Manager writes the final text for each trigger.
# Templates are filled from the facts, so they pass the same number check.

_TEMPLATES: dict[TriggerCode, tuple[str, str, list[tuple[str, str]]]] = {
    TriggerCode.SCORE_DROP: (
        "Your credit score dropped",
        "Your score moved from {previous_score} to {current_score} between "
        "{previous_snapshot_date} and {current_snapshot_date}, a drop of "
        "{points_dropped} points. Review your report to see what changed.",
        [("View my report", "credit_report")],
    ),
    TriggerCode.LOW_SCORE: (
        "Let's work on your credit score",
        "Your current score is {current_score}, which is below {score_line}. "
        "Reviewing your report is a good first step.",
        [("Improve my score", "improve_score")],
    ),
    TriggerCode.POOR_REPAYMENT: (
        "A note on your repayments",
        "We found {late_or_missed_count} late or missed repayments in the last "
        "{window_days} days and {non_performing_count} non-performing facilities on "
        "your report. Catching up on repayments is a good next step.",
        [("View my repayments", "repayments")],
    ),
}


def build_template_message(facts: TriggerFacts) -> GeneratedMessage:
    title, body, actions = _TEMPLATES[facts.trigger_code]
    return GeneratedMessage(
        title=title,
        body=body.format(**facts.facts),
        actions=[Action(label=label, screen=screen) for label, screen in actions],
    )


# ---------- The writer ----------

class WriteResult(BaseModel):
    message: GeneratedMessage
    source: str  # "llm" or "template"
    flagged_for_review: bool
    attempts: int
    errors: list[str] = []  # why the last failed attempt failed (empty on clean success)
    prompt_version: str
    model_version: str
    input_tokens: int = 0
    output_tokens: int = 0


def generate_message(
    facts: TriggerFacts,
    llm: LLMClient,
    vcfg: ValidatorConfig | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> WriteResult:
    vcfg = vcfg or ValidatorConfig()
    system = build_system_prompt(facts, vcfg)
    errors: list[str] = []
    in_tokens = out_tokens = 0

    for attempt in range(1, max_attempts + 1):
        user = build_user_prompt(facts, errors or None)
        try:
            resp = llm.complete(system, user)
        except Exception as exc:  # timeouts, rate limits, provider errors
            errors = [f"llm_error: {type(exc).__name__}"]
            continue

        in_tokens += resp.input_tokens
        out_tokens += resp.output_tokens
        result = validate_message(resp.text, facts, vcfg)
        if result.ok:
            return WriteResult(
                message=result.message,
                source="llm",
                flagged_for_review=False,
                attempts=attempt,
                prompt_version=PROMPT_VERSION,
                model_version=llm.model_version,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
        errors = result.errors

    return WriteResult(
        message=build_template_message(facts),
        source="template",
        flagged_for_review=True,
        attempts=max_attempts,
        errors=errors,
        prompt_version=PROMPT_VERSION,
        model_version=llm.model_version,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
    )


# ======================================================================
# GEMINI CLIENT (PRD step 6, the real provider)
# ======================================================================
# Implements the LLMClient interface above. The google-genai SDK is imported lazily
# inside __init__, so nothing else in this file needs it installed.
#   pip install google-genai
#   export GEMINI_API_KEY=...      (the SDK also reads GOOGLE_API_KEY)
# Pin an EXACT model version, not a floating alias, so model_version in the saved
# record means something later.

def _first_nonblank(*values: str | None) -> str | None:
    """First value that is not None, empty or whitespace-only (stripped)."""
    for v in values:
        if v is not None and v.strip():
            return v.strip()
    return None


class GeminiClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
        client=None,  # injectable for tests
    ):
        if not model or any(ch.isspace() for ch in model):
            raise ValueError(
                f"Invalid Gemini model id {model!r}. Use the exact id from the Gemini "
                "docs or AI Studio: hyphens, no spaces."
            )
        if client is None:
            from google import genai

            key = _first_nonblank(
                api_key, os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY")
            )
            if key is None:
                raise ValueError(
                    "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in "
                    "this terminal session, or pass api_key=..."
                )
            client = genai.Client(api_key=key)
        self._client = client
        self.model_version = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def complete(self, system: str, user: str) -> LLMResponse:
        config = {
            "system_instruction": system,
            "response_mime_type": "application/json",
            "temperature": self._temperature,
        }
        if self._max_output_tokens is not None:
            # Note: on thinking models, thinking tokens can count toward this limit,
            # so a low value can truncate the JSON. Leave unset unless you need a cap.
            config["max_output_tokens"] = self._max_output_tokens

        resp = self._client.models.generate_content(
            model=self.model_version, contents=user, config=config
        )

        usage = getattr(resp, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = (getattr(usage, "candidates_token_count", 0) or 0) + (
            getattr(usage, "thoughts_token_count", 0) or 0
        )
        # resp.text is None when the answer was blocked or empty. Returning "" makes
        # validation fail, which triggers the normal retry and template fallback.
        return LLMResponse(
            text=resp.text or "",
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )