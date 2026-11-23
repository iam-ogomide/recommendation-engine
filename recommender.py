"""CreditCliq Recommendation Agent, in one file.

    facts in  ->  prompt  ->  ONE model call  ->  checks  ->  (retry once)  ->  Draft out
                                                          \\->  template wording + review flag

The orchestrator decides WHO hears from us and WHY (trigger rules, sending limits). This file only
decides HOW a fired trigger is worded. It never picks a recipient, never calls a tool, never loops
on its own, and never sees a name, BVN, phone or account number: the input type has no field for one.

Read it top to bottom:
  1. SETTINGS    every value a non-programmer may want to change (templates, wording rules, screens)
  2. CONTRACT    TriggerFacts in, Draft out
  3. MODEL SEAM  the only door to a model (swap in a gateway or local model here)
  4. CHECKS      the guardrails: plain code, no model
  5. THE AGENT   Recommender.write()
  6. SELF-TEST   python recommender.py --selftest      (no model, no network)

Run:
  pip install pydantic            # and google-genai for the real model
  python recommender.py --selftest
  python recommender.py --demo
  GEMINI_API_KEY=... python recommender.py --live --model <exact-model-id>
  python recommender.py --schemas  # prints the JSON contract to share with the trigger owners
"""
from __future__ import annotations
import argparse
import difflib
from collections import Counter
import hashlib
import json
import os
import re
import sys
from datetime import date
from enum import Enum
from string import Template
from typing import Annotated, Literal, Protocol, Union
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator
from dotenv import load_dotenv
load_dotenv()  

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 



# =============================================================================
# 1. SETTINGS  
# =============================================================================

# ---- limits ------------------------------------------------------------------
MAX_ATTEMPTS = 2            # first try plus ONE retry (design: one retry, then the template)
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = None    # leave None on thinking models: thinking tokens count toward the cap
SEND_RESPONSE_SCHEMA = False  # send the output JSON schema to the model. UNTESTED on live Gemini
 
# PRD message limits (product rules)
TITLE_MAX = 60
BODY_MAX = 400
ACTIONS_MAX = 3
 
# ---- bureau names shown to the subscriber (facts carry the canonical code) -----
BUREAU_DISPLAY = {"FIRST_CENTRAL": "FirstCentral", "CRC": "CRC", "CREDIT_REGISTRY": "Credit Registry"}
 
# ---- screens an action may open, per trigger. PLACEHOLDERS: get the real ids from Mobile ----
ALLOWED_SCREENS = {
    "SCORE_DROP": ("credit_report", "improve_score"),
    "LOW_SCORE": ("improve_score", "credit_report"),
    "POOR_REPAYMENT": ("repayments", "improve_score"),
}
 
# ---- fallback wording. PLACEHOLDERS: the Product Manager writes the final text. -----------
# Placeholders are the fact names, plus {bureau_name}. Every template is checked at start-up.
TEMPLATES = {
    "SCORE_DROP": {
        "title": "Your credit score dropped",
        "body": "Your {bureau_name} score moved from {previous_score} to {current_score} between "
                "{previous_snapshot_date} and {current_snapshot_date}. Points lost: {points_dropped}. "
                "Review your report to see what changed.",
        "actions": [("View my report", "credit_report")],
    },
    "LOW_SCORE": {
        "title": "Let's work on your credit score",
        "body": "Your {bureau_name} score is {current_score}, which is below {score_line}. "
                "Reviewing your report is a good first step.",
        "actions": [("Improve my score", "improve_score")],
    },
    "POOR_REPAYMENT": {
        "title": "A note on your repayments",
        "body": "Late or missed repayments in the last {window_days} days: {late_or_missed_count}. "
                "Non-performing facilities on your {bureau_name} report: {non_performing_count}. "
                "Catching up on repayments is a good next step.",
        "actions": [("View my repayments", "repayments")],
    },
}
 
# ---- wording rules (PRD "What a message may say"). STARTER lists: compliance should extend them,
#      and every new rule needs a red-team case in the self-test. ------------------------------
BANNED_PHRASES = ("guarantee", "will increase your score", "approved")
BANNED_PATTERNS = {
    "PROMISE": (
        r"\bwill (?:increase|improve|raise|boost|lift|recover|go up|rise)\b",
        r"\byou(?:'ll| will) (?:qualify|get (?:a )?loan|be (?:eligible|accepted))\b",
        r"\bapproval\b",
        r"\bwithin (?:the next |a |just a )?(?:few )?(?:days|weeks|months)\b",
    ),
    "BUREAU_METHOD": (
        r"\balgorithm", r"\bformula\b", r"\bcalculat(?:e|es|ed|ion)\b",
        r"\bweight(?:s|ed|ing)?\b", r"\bscoring (?:model|method|criteria)\b", r"\bdetermined by\b",
    ),
    "SPECULATED_CAUSE": (
        r"\bdue to\b", r"\bcaused by\b", r"\bas a result of\b", r"\bloan application",
        r"\bhard (?:pull|inquiry|enquiry)\b", r"\b(?:inquiry|inquiries|enquiry|enquiries)\b",
    ),
    "ADVICE": (
        r"\b(?:lawyer|attorney|solicitor|lawsuit|litigation|court)\b",
        r"\blegal (?:action|advice|rights)\b",
        r"\b(?:tax|taxes|taxation|firs|vat)\b",
        r"\b(?:invest|investing|investment|investments|stocks?|securities|crypto|bitcoin)\b",
        r"\b(?:mutual funds?|treasury bills?|bankruptcy|insolvency)\b",
    ),
}
LENDER_NAMES = (
    "GTBank", "GT Bank", "Guaranty Trust", "Access Bank", "Zenith", "UBA", "United Bank for Africa",
    "First Bank", "FirstBank", "FCMB", "Fidelity Bank", "Sterling Bank", "Wema", "Union Bank", "Polaris",
    "Stanbic", "Ecobank", "Opay", "Palmpay", "Kuda", "Moniepoint", "FairMoney", "Renmoney",
)
RULE_HINTS = {  # what the retry prompt says when a wording rule fires
    "PROMISE": "do not promise a higher score, an approval or a timeline",
    "BUREAU_METHOD": "do not describe how a bureau calculates its score",
    "SPECULATED_CAUSE": "do not guess why the score changed",
    "ADVICE": "do not give legal, tax or investment advice",
    "LENDER": "do not name any lender",
}
 
# ---- the prompt. Bump PROMPT_VERSION whenever you change any of it. Record it in your changelog
#      and rerun the evals. The hash below also changes, so a silent edit is detectable. -----------
PROMPT_VERSION = "write_v2"
 
SYSTEM_PROMPT = Template("""\
You write short in-app recommendations for CreditCliq subscribers about their credit report.
 
Reply with ONLY a JSON object, no markdown and no code fences:
{"title": string (max $title_max characters), "body": string (max $body_max characters), "actions": [{"label": string, "screen": string}] (0 to $max_actions items)}
 
Rules:
- Open the body with the reason the subscriber is hearing from us, in plain words.
- Use only numbers that appear in the facts, written as digits. Never write a number as a word. Never compute or invent a number.
- Show every number the guidance below asks for.
- Dates may be written in a friendly form, using only the day numbers that appear in the facts. Write the day without a leading zero: "5 Jul", never "05 Jul".
- Use correct singular and plural: 1 point, 2 points; 1 repayment, 2 repayments; 1 facility, 2 facilities.
- Never state or work out a score other than the ones in the facts.
- Never promise a higher score, an approval or a timeline. Never use the words "guarantee" or "approved".
- Never explain how a bureau calculates its score, and never guess why a score changed.
- Never mention other people or lenders, and never give legal, tax or investment advice.
- Never include a BVN, account number, phone number or email address.
- Each action's screen must be one of: $screens.
Tone: warm, clear and short. Name the bureau as given in the user message.""")
 
TRIGGER_GUIDANCE = {
    "SCORE_DROP": "Say the score dropped. Show the previous score, the current score and the number of points "
                  "dropped, in the right order (previous first, then current). Mention both dates. Do not "
                  "guess a cause. Do not say the score rose or improved.",
    "LOW_SCORE": "Say the current score is below the score line. Show both numbers. Encourage practical next "
                 "steps without promising any result.",
    "POOR_REPAYMENT": "Say how many late or missed repayments were found and over what window, and how many "
                      "non-performing facilities are on the report. Only mention a count that is not zero. "
                      "Late or missed repayments come from the subscriber's own repayment history, NOT from the "
                      "bureau report: never say they are on the report. Only non-performing facilities are on the "
                      "bureau report. Encourage catching up on repayments.",
}
 
# One worked example per trigger, shown to the model. The self-test checks that each example passes
# this file's own checks, so the prompt never teaches something the guardrails would reject.
PROMPT_EXAMPLES = {
    "SCORE_DROP": (
        {"bureau": "FIRST_CENTRAL", "previous_score": 710, "current_score": 695, "points_dropped": 15,
         "previous_snapshot_date": "2026-08-14", "current_snapshot_date": "2026-09-14"},
        {"title": "Your credit score dropped",
         "body": "Your FirstCentral score moved from 710 to 695 between 14 Aug and 14 Sep, a drop of 15 points. "
                 "Open your report to see what is on it, then pick a next step below.",
         "actions": [{"label": "View my report", "screen": "credit_report"},
                     {"label": "Improve my score", "screen": "improve_score"}]},
    ),
    "LOW_SCORE": (
        {"bureau": "FIRST_CENTRAL", "current_score": 572, "score_line": 600},
        {"title": "Let's work on your credit score",
         "body": "Your FirstCentral score is 572, which is below 600. Small, steady steps can help over time, "
                 "and your report is a good place to start.",
         "actions": [{"label": "Improve my score", "screen": "improve_score"}]},
    ),
    "POOR_REPAYMENT": (
        {"bureau": "FIRST_CENTRAL", "late_or_missed_count": 3, "non_performing_count": 1, "window_days": 90},
        {"title": "A note on your repayments",
         "body": "We found 3 late or missed repayments in your repayment history over the last 90 days, and 1 "
                 "non-performing facility on your FirstCentral report. Catching up on repayments is a good next step.",
         "actions": [{"label": "View my repayments", "screen": "repayments"}]},
    ),
}
 
 
def _guidance_text(trigger: str) -> str:
    facts, answer = PROMPT_EXAMPLES[trigger]
    return (f"{TRIGGER_GUIDANCE[trigger]}\nEXAMPLE_FACTS: {json.dumps(facts)}\n"
            f"EXAMPLE_ANSWER: {json.dumps(answer)}")
 
 
PROMPT_HASH = hashlib.sha256(
    (SYSTEM_PROMPT.template + "".join(_guidance_text(t) for t in sorted(TRIGGER_GUIDANCE))).encode()
).hexdigest()[:12]  # changes if ANY prompt text changes, even without a version bump
 
 
# =============================================================================
# 2. CONTRACT: TriggerFacts in, Draft out
# =============================================================================
# The agent only ever sees numbers, dates and a bureau code. There is NO field for a subscriber id,
# name, BVN, phone or account number, and unknown fields are rejected (extra="forbid"), so an
# identifier cannot slip in by accident. Facts must also be internally consistent: if they are not,
# something upstream is broken and we refuse to write a confident message about wrong numbers.
 
class TriggerCode(str, Enum):
    SCORE_DROP = "SCORE_DROP"
    LOW_SCORE = "LOW_SCORE"
    POOR_REPAYMENT = "POOR_REPAYMENT"
 
 
Score = Annotated[int, Field(ge=0, le=1000)]
 
 
class _Facts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    bureau: str = Field(min_length=1)  # canonical code, e.g. FIRST_CENTRAL
 
    def payload(self) -> dict[str, int | str]:
        """The facts exactly as sent to the writer (also stored as `trigger_facts`)."""
        return self.model_dump(mode="json", exclude={"trigger_code"})
 
    def required_numbers(self) -> list[int]:
        raise NotImplementedError
 
 
class ScoreDropFacts(_Facts):
    trigger_code: Literal["SCORE_DROP"] = "SCORE_DROP"
    previous_score: Score
    current_score: Score
    points_dropped: int = Field(ge=1)
    previous_snapshot_date: date
    current_snapshot_date: date
 
    @model_validator(mode="after")
    def _consistent(self):
        if self.previous_score - self.current_score != self.points_dropped:
            raise ValueError("points_dropped must equal previous_score - current_score")
        if self.previous_snapshot_date >= self.current_snapshot_date:
            raise ValueError("previous_snapshot_date must be before current_snapshot_date")
        return self
 
    def required_numbers(self) -> list[int]:
        return [self.previous_score, self.current_score]
 
 
class LowScoreFacts(_Facts):
    trigger_code: Literal["LOW_SCORE"] = "LOW_SCORE"
    current_score: Score
    score_line: Score
 
    @model_validator(mode="after")
    def _below_line(self):
        if self.current_score >= self.score_line:
            raise ValueError("LOW_SCORE facts need current_score below score_line")
        return self
 
    def required_numbers(self) -> list[int]:
        return [self.current_score, self.score_line]
 
 
class PoorRepaymentFacts(_Facts):
    trigger_code: Literal["POOR_REPAYMENT"] = "POOR_REPAYMENT"
    late_or_missed_count: int = Field(ge=0)
    non_performing_count: int = Field(ge=0)
    window_days: int = Field(ge=1)
 
    @model_validator(mode="after")
    def _something_to_say(self):
        if self.late_or_missed_count == 0 and self.non_performing_count == 0:
            raise ValueError("POOR_REPAYMENT facts need at least one non-zero count")
        return self
 
    def required_numbers(self) -> list[int]:
        # Only counts that are not zero: we never force "0 non-performing" into a message.
        return [n for n in (self.late_or_missed_count, self.non_performing_count) if n > 0]
 
 
Facts = Annotated[Union[ScoreDropFacts, LowScoreFacts, PoorRepaymentFacts], Field(discriminator="trigger_code")]
_FACTS_ADAPTER: TypeAdapter = TypeAdapter(Facts)
 
 
def parse_facts(data: dict | _Facts) -> _Facts:
    """Validate raw facts from the orchestrator. Raises pydantic.ValidationError."""
    return data if isinstance(data, _Facts) else _FACTS_ADAPTER.validate_python(data)
 
 
class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    screen: str
 
 
class Message(BaseModel):
    # The shape the MODEL must return.
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=TITLE_MAX)
    body: str = Field(min_length=1, max_length=BODY_MAX)
    actions: list[Action] = Field(default_factory=list, max_length=ACTIONS_MAX)
 
 
class Origin(str, Enum):
    MODEL = "model"        # written by the model and passed every check
    TEMPLATE = "template"  # the PM's fallback wording
 
 
class Violation(BaseModel):
    """One failed check. `code` is stable (safe to count in logs); `detail` is for humans and the
    retry prompt. It never repeats a BVN-like or phone-like string."""
    code: str
    detail: str = ""
 
 
class Draft(BaseModel):
    trigger_code: TriggerCode
    trigger_facts: dict[str, int | str]  # exactly what was sent to the writer
    title: str
    body: str
    actions: list[Action]
    origin: Origin
    needs_review: bool                   # True: a human must read it before any subscriber does
    review_reasons: list[Violation] = Field(default_factory=list)  # the LAST attempt's violations (why it needs review)
    failed_attempt_codes: list[list[str]] = Field(default_factory=list)  # codes from EVERY rejected attempt, in order
    attempts: int
    prompt_version: str
    prompt_hash: str
    model_version: str | None = None     # None if no model call succeeded
    input_tokens: int = 0
    output_tokens: int = 0
 
    @property
    def publishable(self) -> bool:
        """The Manager must never auto-publish a draft that needs review."""
        return not self.needs_review
 
 
# =============================================================================
# 3. MODEL SEAM: the only door to a model
# =============================================================================
# Today: a direct Gemini client. Later: a client for the model gateway (identifier scan, token caps,
# routing to template / local model / external model, cost metering). The agent only needs
# `complete()`, so that swap changes nothing else in this file.
 
class LLMResponse(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
 
 
class LLMClient(Protocol):
    model_version: str  # pin an EXACT version, never a floating alias
 
    def complete(self, system: str, user: str, *, max_output_tokens: int | None = None,
                 json_schema: dict | None = None) -> LLMResponse: ...
 
 
class ScriptedLLM:
    """A fake model for tests and demos: returns queued answers in order; an Exception in the queue
    is raised instead. Records every call so tests can inspect the prompts."""
    model_version = "scripted-1"
 
    def __init__(self, answers: list):
        self._answers = list(answers)
        self.calls: list[dict] = []
 
    def complete(self, system, user, *, max_output_tokens=None, json_schema=None) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        if not self._answers:
            raise RuntimeError("ScriptedLLM ran out of answers")
        item = self._answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item, input_tokens=100, output_tokens=60)
 
 
class GeminiClient:
    """pip install google-genai ; export GEMINI_API_KEY=... ; pin an exact model id."""
 
    def __init__(self, model: str, api_key: str | None = None, temperature: float = TEMPERATURE, client=None):
        if not model or any(ch.isspace() for ch in model):
            raise ValueError(f"Invalid Gemini model id {model!r}: use the exact id, no spaces.")
        if client is None:
            from google import genai  # imported lazily so the rest works without the SDK
            key = next((k.strip() for k in (api_key, os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY"))
                        if k and k.strip()), None)
            if key is None:
                raise ValueError("No Gemini API key. Set GEMINI_API_KEY or pass api_key=...")
            client = genai.Client(api_key=key)
        self._client = client
        self.model_version = model
        self._temperature = temperature
 
    def complete(self, system, user, *, max_output_tokens=None, json_schema=None) -> LLMResponse:
        config: dict = {"system_instruction": system, "response_mime_type": "application/json",
                        "temperature": self._temperature}
        if max_output_tokens is not None:
            config["max_output_tokens"] = max_output_tokens  # thinking tokens count toward this cap
        if json_schema is not None:
            config["response_json_schema"] = json_schema     # UNTESTED on the live API
        resp = self._client.models.generate_content(model=self.model_version, contents=user, config=config)
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = (getattr(usage, "candidates_token_count", 0) or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
        # resp.text is None when blocked or empty; "" fails the format check -> normal retry/fallback
        return LLMResponse(text=resp.text or "", input_tokens=in_tok, output_tokens=out_tok)
 
 
# =============================================================================
# 4. CHECKS: the guardrails. Pure code, no model, no network.
# =============================================================================
# check_output() returns EVERY violation (not just the first), each with a stable code. The agent
# feeds the details back to the model on the retry, and the self-test uses the same function.
#
# CAN catch: format and length, numbers, required facts, number order, direction words, banned
# wording, PII shapes, allowed screens.
# CANNOT catch: meaning. A paraphrase of a banned idea can slip through, and a sentence can be
# true-looking but misleading. For that you need the red-team cases (grow them), Admin review in the
# first months (PRD), and later an independent Policy Reviewer.
 
FORMAT_NOT_JSON = "FORMAT_NOT_JSON"
FORMAT_SCHEMA = "FORMAT_SCHEMA"            # wrong shape, too long, too many actions, extra fields
EMPTY_TEXT = "EMPTY_TEXT"
ACTION_INCOMPLETE = "ACTION_INCOMPLETE"
NUMBER_NOT_IN_FACTS = "NUMBER_NOT_IN_FACTS"
NUMBER_WORD = "NUMBER_WORD"
REQUIRED_FACT_MISSING = "REQUIRED_FACT_MISSING"
NUMBER_ORDER = "NUMBER_ORDER"              # "from 695 to 710" when the score went 710 -> 695
POINTS_MISMATCH = "POINTS_MISMATCH"        # "a drop of 710 points"
WRONG_DIRECTION = "WRONG_DIRECTION"        # "rose", "improved" in a SCORE_DROP message
BANNED_PHRASE = "BANNED_PHRASE"
LENDER = "LENDER"
PII = "PII"
SCREEN_NOT_ALLOWED = "SCREEN_NOT_ALLOWED"
PLURAL = "PLURAL"                          # "1 points", "2 repayment"
DATE_FORMAT = "DATE_FORMAT"                # "05 Jul" (leading zero)
SOURCE_MISATTRIBUTION = "SOURCE_MISATTRIBUTION"  # late repayments said to be "on the bureau report"
# wording-rule codes are the BANNED_PATTERNS keys: PROMISE, BUREAU_METHOD, SPECULATED_CAUSE, ADVICE
 
 
class CheckResult(BaseModel):
    ok: bool
    message: Message | None = None
    violations: list[Violation] = []
 
    @property
    def codes(self) -> set[str]:
        return {v.code for v in self.violations}
 
 
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_LONG_DIGITS = re.compile(r"\d{10,}")  # BVN (11), account no. (10), bare phone numbers
_NG_PHONE = re.compile(r"(\+?234[\s-]?|\b0)[789][01]\d[\s-]?\d{3}[\s-]?\d{4}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Number WORDS slip past a digit check ("two late payments"). "one" is flagged only when it clearly
# counts something, because "one step" is harmless.
_NUMBER_WORDS = re.compile(
    r"\b(?:zero|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    r"thousand|twice|double|triple|half)\b", re.I)
_ONE_COUNTING = re.compile(r"\bone\s+(?:late|missed|non-?performing|repayment|facility|facilities|point)\b", re.I)
_FROM_TO = re.compile(r"\bfrom\s+(\d[\d,]*)\s+to\s+(\d[\d,]*)", re.I)
_N_POINTS = re.compile(r"(\d[\d,]*)\s*-?\s*points?\b", re.I)
_WRONG_DIRECTION = re.compile(r"\b(?:rose|risen|increased|went up|gone up|improved|gained)\b", re.I)
_MOD = r"(?:(?:late or missed|late|missed|non-performing)\s+)?"  # an optional modifier between number and noun
_ONE_PLURAL = re.compile(r"\b1\s+" + _MOD + r"(?:points|repayments|facilities|days)\b", re.I)
_MANY_SINGULAR = re.compile(r"\b(?:[02-9]|\d{2,})\s+" + _MOD + r"(?:point|repayment|facility|day)\b", re.I)
_LEADING_ZERO_DATE = re.compile(r"\b0\d\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b", re.I)
_LATE_WORD = re.compile(r"\b(?:late|missed)\b", re.I)
_REPORT_WORD = re.compile(r"\b(?:report|bureau|firstcentral|crc|credit registry|credit (?:record|file|profile|history))\b", re.I)
_NONPERF_WORD = re.compile(r"non-?performing", re.I)
 
 
def numbers_in(text: str) -> set[float]:
    return {float(m.replace(",", "")) for m in _NUMBER.findall(text)}
 
 
def allowed_numbers(payload: dict) -> set[float]:
    """Every number that may appear: integers in the facts, plus the digits of dates."""
    allowed: set[float] = set()
    for value in payload.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            allowed.add(float(abs(value)))
        else:  # e.g. an ISO date 2026-10-10 -> 2026, 10, 10
            allowed |= numbers_in(str(value))
    return allowed
 
 
def _f(s: str) -> float:
    return float(s.replace(",", ""))
 
 
def parse_message(raw: str) -> tuple[Message | None, list[Violation]]:
    """Is it valid JSON in the agreed shape and within the length limits?"""
    try:
        data = json.loads(_FENCE.sub("", raw.strip()))
    except json.JSONDecodeError:
        return None, [Violation(code=FORMAT_NOT_JSON, detail="output is not valid JSON")]
    try:
        return Message.model_validate(data), []
    except ValidationError as exc:
        details = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return None, [Violation(code=FORMAT_SCHEMA, detail="; ".join(details))]
 
 
def check_output(raw: str, facts: _Facts) -> CheckResult:
    msg, problems = parse_message(raw)
    if msg is None:
        return CheckResult(ok=False, violations=problems)
 
    v: list[Violation] = []
    trigger, payload = facts.trigger_code, facts.payload()
 
    if not msg.title.strip() or not msg.body.strip():
        v.append(Violation(code=EMPTY_TEXT, detail="title and body must not be empty"))
    if any(not a.label.strip() or not a.screen.strip() for a in msg.actions):
        v.append(Violation(code=ACTION_INCOMPLETE, detail="every action needs a label and a screen"))
 
    prose = f"{msg.title} {msg.body}"
    text = " ".join([msg.title, msg.body, *(a.label for a in msg.actions)])
 
    # numbers: every one must come from the facts (never echo a long number: it may be a BVN)
    extra = sorted(numbers_in(text) - allowed_numbers(payload))
    if extra:
        shown = ["a very long number" if n >= 1e9 else f"{n:g}" for n in extra]
        v.append(Violation(code=NUMBER_NOT_IN_FACTS, detail="numbers not in the facts: " + ", ".join(shown)))
 
    if _NUMBER_WORDS.search(text) or _ONE_COUNTING.search(text):
        v.append(Violation(code=NUMBER_WORD, detail="write numbers as digits, not words"))
 
    # required facts must actually be shown
    shown_nums = numbers_in(prose)
    missing = [n for n in facts.required_numbers() if float(n) not in shown_nums]
    if missing:
        v.append(Violation(code=REQUIRED_FACT_MISSING,
                           detail="the message must show these numbers from the facts: " + ", ".join(map(str, missing))))
 
    # score-drop specifics: order, points, direction
    if trigger == "SCORE_DROP":
        prev, cur, dropped = payload["previous_score"], payload["current_score"], payload["points_dropped"]
        for a, b in _FROM_TO.findall(prose):
            if (_f(a), _f(b)) != (float(prev), float(cur)):
                v.append(Violation(code=NUMBER_ORDER, detail=f"the score went from {prev} to {cur}, not from {a} to {b}"))
                break
        for n in _N_POINTS.findall(prose):
            if _f(n) != float(dropped):
                v.append(Violation(code=POINTS_MISMATCH, detail=f"the number of points dropped is {dropped}, not {n}"))
                break
        if _WRONG_DIRECTION.search(prose):
            v.append(Violation(code=WRONG_DIRECTION, detail="the score dropped: do not say it rose or improved"))
 
    # banned wording
    lowered = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase.lower() in lowered:
            v.append(Violation(code=BANNED_PHRASE, detail=f"banned phrase: {phrase}"))
    for category, patterns in BANNED_PATTERNS.items():
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                v.append(Violation(code=category, detail=f"{RULE_HINTS.get(category, 'remove this wording')} (matched {m.group(0)!r})"))
                break  # one report per category is enough
    m = _LENDER_RE.search(text)
    if m:
        v.append(Violation(code=LENDER, detail=f"{RULE_HINTS['LENDER']} (matched {m.group(0)!r})"))
 
    # PII shapes. The detail never repeats the matched text.
    if _LONG_DIGITS.search(text) or _NG_PHONE.search(text) or _EMAIL.search(text):
        v.append(Violation(code=PII, detail="text contains something that looks like a BVN, account number, phone number or email"))
 
    # grammar and formatting the model gets wrong even when every number is right
    if _ONE_PLURAL.search(prose) or _MANY_SINGULAR.search(prose):
        v.append(Violation(code=PLURAL, detail="use the right singular/plural: 1 point, 2 points; 1 repayment, 2 repayments"))
    if _LEADING_ZERO_DATE.search(prose):
        v.append(Violation(code=DATE_FORMAT, detail='write dates without a leading zero: "5 Jul", not "05 Jul"'))
 
    # late or missed repayments come from the subscriber's repayment history, not from the bureau report
    if trigger == "POOR_REPAYMENT":
        for sentence in re.split(r"(?<=[.!?])\s+", prose):
            if _LATE_WORD.search(sentence) and _REPORT_WORD.search(sentence) and not _NONPERF_WORD.search(sentence):
                v.append(Violation(code=SOURCE_MISATTRIBUTION,
                                   detail="late or missed repayments come from the repayment history, not the bureau report; "
                                          "only non-performing facilities are on the report"))
                break
 
    # action screens must be on the allowed list for this trigger
    for a in msg.actions:
        if a.screen not in ALLOWED_SCREENS[trigger]:
            v.append(Violation(code=SCREEN_NOT_ALLOWED, detail=f"action screen not allowed: {a.screen}"))
 
    return CheckResult(ok=False, violations=v) if v else CheckResult(ok=True, message=msg)
 
 
_LENDER_RE = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in LENDER_NAMES) + r")\b", re.IGNORECASE)
 
 
# =============================================================================
# 5. THE AGENT
# =============================================================================
 
def bureau_name(code: str) -> str:
    return BUREAU_DISPLAY.get(code, code.replace("_", " ").title())
 
 
class Recommender:
    def __init__(self, llm: LLMClient, max_attempts: int = MAX_ATTEMPTS,
                 send_response_schema: bool = SEND_RESPONSE_SCHEMA):
        self._llm = llm
        self._max_attempts = max_attempts
        self._schema = Message.model_json_schema() if send_response_schema else None
 
    # ---- prompts ----
    def _system_prompt(self, trigger: str) -> str:
        return SYSTEM_PROMPT.substitute(screens=", ".join(ALLOWED_SCREENS[trigger]),
                                        title_max=TITLE_MAX, body_max=BODY_MAX, max_actions=ACTIONS_MAX)
 
    def _user_prompt(self, facts: _Facts, previous: list[Violation] | None) -> str:
        # Fixed text FIRST (same for every call of this trigger) so a local model server can reuse its cached
        # start of the prompt; the facts that change come last.
        parts = [
            _guidance_text(facts.trigger_code),
            f"Trigger: {facts.trigger_code}",
            f"Name the bureau as: {bureau_name(facts.bureau)}",
            f"Facts (JSON): {json.dumps(facts.payload(), sort_keys=True)}",
        ]
        if previous:
            parts.append("Your previous answer was rejected for these reasons. Fix them and answer again:\n- "
                         + "\n- ".join(x.detail or x.code for x in previous))
        return "\n".join(parts)
 
    # ---- the template fallback ----
    def _template_parts(self, facts: _Facts) -> tuple[str, str, list[Action]]:
        t = TEMPLATES[facts.trigger_code]
        context = {**facts.payload(), "bureau_name": bureau_name(facts.bureau)}
        return t["title"], t["body"].format(**context), [Action(label=l, screen=s) for l, s in t["actions"]]
 
    # ---- the one public method ----
    def write(self, facts: _Facts | dict) -> Draft:
        """Write one recommendation. Never raises because of the model: any failure ends in the
        template wording with needs_review=True. Raises pydantic.ValidationError only if the FACTS
        are invalid (an upstream bug, not a model problem)."""
        facts = parse_facts(facts)
        trigger = TriggerCode(facts.trigger_code)
        system = self._system_prompt(trigger.value)
 
        violations: list[Violation] = []
        failed: list[list[str]] = []  # codes of every rejected attempt (the audit trail of WHY a retry happened)
        in_tok = out_tok = 0
        model_version: str | None = None
 
        for attempt in range(1, self._max_attempts + 1):
            try:
                resp = self._llm.complete(system, self._user_prompt(facts, violations or None),
                                          max_output_tokens=MAX_OUTPUT_TOKENS, json_schema=self._schema)
            except Exception as exc:  # timeouts, rate limits, provider errors: log the TYPE only, never the message
                violations = [Violation(code="LLM_ERROR", detail=type(exc).__name__)]
                failed.append(["LLM_ERROR"])
                continue
            model_version = self._llm.model_version
            in_tok += resp.input_tokens
            out_tok += resp.output_tokens
            result = check_output(resp.text, facts)
            if result.ok:
                m = result.message
                return self._draft(facts, trigger, m.title, m.body, m.actions, Origin.MODEL, False, [], failed,
                                   attempt, model_version, in_tok, out_tok)
            violations = result.violations
            failed.append(list(dict.fromkeys(v.code for v in violations)))  # unique, in order
 
        title, body, actions = self._template_parts(facts)
        return self._draft(facts, trigger, title, body, actions, Origin.TEMPLATE, True, violations, failed,
                           self._max_attempts, model_version, in_tok, out_tok)
 
    @staticmethod
    def _draft(facts, trigger, title, body, actions, origin, needs_review, reasons, failed, attempts,
               model_version, in_tok, out_tok) -> Draft:
        return Draft(trigger_code=trigger, trigger_facts=facts.payload(), title=title, body=body,
                     actions=actions, origin=origin, needs_review=needs_review, review_reasons=reasons,
                     failed_attempt_codes=failed, attempts=attempts, prompt_version=PROMPT_VERSION, prompt_hash=PROMPT_HASH,
                     model_version=model_version, input_tokens=in_tok, output_tokens=out_tok)
 
 
# ---- start-up validation: a bad edit to the SETTINGS block fails here, not at 02:00 in a run ----
def _validate_settings() -> None:
    codes = {c.value for c in TriggerCode}
    for name, table in (("ALLOWED_SCREENS", ALLOWED_SCREENS), ("TEMPLATES", TEMPLATES),
                        ("TRIGGER_GUIDANCE", TRIGGER_GUIDANCE), ("PROMPT_EXAMPLES", PROMPT_EXAMPLES)):
        if set(table) != codes:
            raise RuntimeError(f"{name} must have exactly the triggers {sorted(codes)}")
    for patterns in BANNED_PATTERNS.values():
        for p in patterns:
            re.compile(p)  # raises re.error on a bad regex
    for code, t in TEMPLATES.items():
        if len(t["title"]) > TITLE_MAX or len(t["actions"]) > ACTIONS_MAX:
            raise RuntimeError(f"TEMPLATES[{code}] breaks the title or action limit")
        for _, screen in t["actions"]:
            if screen not in ALLOWED_SCREENS[code]:
                raise RuntimeError(f"TEMPLATES[{code}] uses screen {screen!r} that is not in ALLOWED_SCREENS")
 
 
_validate_settings()
 
 
# =============================================================================
# 6. SELF-TEST  (python recommender.py --selftest)
# =============================================================================
# Two kinds of case. GOLDEN: good messages that MUST be accepted (catches false rejections).
# RED-TEAM: bad messages that MUST be rejected, for a specific reason (catches holes in the checks).
# Whenever you meet a new failure in real life, add a red-team case for it here.
 
def _sd(prev, cur, d1, d2, b="FIRST_CENTRAL"):
    return {"trigger_code": "SCORE_DROP", "bureau": b, "previous_score": prev, "current_score": cur,
            "points_dropped": prev - cur, "previous_snapshot_date": d1, "current_snapshot_date": d2}
 
 
def _ls(cur, line=600, b="FIRST_CENTRAL"):
    return {"trigger_code": "LOW_SCORE", "bureau": b, "current_score": cur, "score_line": line}
 
 
def _pr(late, npl, win=90, b="FIRST_CENTRAL"):
    return {"trigger_code": "POOR_REPAYMENT", "bureau": b, "late_or_missed_count": late,
            "non_performing_count": npl, "window_days": win}
 
 
def _out(title, body, actions):
    return json.dumps({"title": title, "body": body, "actions": [{"label": l, "screen": s} for l, s in actions]})
 
 
_RPT, _IMP, _REP = [("View my report", "credit_report")], [("Improve my score", "improve_score")], [("View my repayments", "repayments")]
_D, _L, _P = _sd(710, 695, "2026-08-14", "2026-09-14"), _ls(572), _pr(2, 1)
_BASE_D = "Your FirstCentral score moved from 710 to 695 between 14 Aug and 14 Sep, a drop of 15 points."
_BASE_L = "Your FirstCentral score is 572, which is below 600."
_BASE_P = "We found 2 late or missed repayments in the last 90 days and 1 non-performing facility."
 
GOLDEN = [  # (id, facts, good output)
    ("drop-typical", _D, _out("Your credit score dropped", _BASE_D + " Open your report to see what is on it.", _RPT)),
    ("drop-one-point", _sd(650, 649, "2026-09-20", "2026-10-20"),
     _out("Your credit score dropped", "Your FirstCentral score slipped from 650 to 649, a drop of 1 point, since your last check on 20 Sep.", _RPT)),
    ("drop-large", _sd(780, 702, "2026-07-05", "2026-10-05"),
     _out("Your FirstCentral score fell 78 points", "Your score went from 780 on 5 Jul to 702 on 5 Oct, a drop of 78 points. Open your report and review what is listed.", _RPT)),
    ("low-typical", _L, _out("Let's work on your credit score", _BASE_L + " Your report is a good place to start.", _IMP)),
    ("low-599", _ls(599), _out("A look at your credit score", "Your FirstCentral score is 599, which is below 600. Pick one small step to work on this month.", _IMP)),
    ("repay-both", _pr(3, 1), _out("A note on your repayments", "We found 3 late or missed repayments in your repayment history over the last 90 days, and 1 non-performing facility on your FirstCentral report. Catching up is a good next step.", _REP)),
    ("repay-history-source", _pr(3, 0), _out("A note on your repayments", "We found 3 late or missed repayments in your repayment history over the last 90 days. Catching up is a good next step.", _REP)),
    ("repay-late-only", _pr(2, 0), _out("A note on your repayments", "We found 2 late or missed repayments in the last 90 days. Catching up on repayments is a good next step.", _REP)),
    ("repay-npl-only", _pr(0, 2), _out("A note on your report", "We found 2 non-performing facilities on your FirstCentral report. Reviewing your repayments is a good next step.", _REP)),
]
 
# HELD-OUT: facts the prompt never shows and GOLDEN does not reuse. The live run uses these to see whether
# the model can WRITE for new numbers, or is only copying the prompt's worked examples. No expected output:
# the checks are the judge (and you, reading the messages with --show).
HELD_OUT = [  # (id, facts)
    ("h-drop-2pts", _sd(742, 740, "2026-06-11", "2026-07-11")),
    ("h-drop-9-crc", _sd(698, 689, "2026-07-18", "2026-08-18", "CRC")),
    ("h-drop-12-crc", _sd(588, 576, "2026-09-03", "2026-10-03", "CRC")),
    ("h-drop-35", _sd(655, 620, "2026-08-29", "2026-09-29")),
    ("h-drop-100", _sd(820, 715, "2026-04-30", "2026-05-30")),
    ("h-drop-low-range", _sd(420, 399, "2026-10-01", "2026-11-01")),
    ("h-drop-year-cross", _sd(603, 597, "2026-12-15", "2027-01-15", "CREDIT_REGISTRY")),
    ("h-low-305", _ls(305)),
    ("h-low-448-registry", _ls(448, 600, "CREDIT_REGISTRY")),
    ("h-low-512", _ls(512)),
    ("h-low-590-crc", _ls(590, 600, "CRC")),
    ("h-low-597", _ls(597)),
    ("h-repay-4-0", _pr(4, 0)),
    ("h-repay-0-1", _pr(0, 1)),
    ("h-repay-3-0-registry", _pr(3, 0, 90, "CREDIT_REGISTRY")),
    ("h-repay-7-2", _pr(7, 2)),
    ("h-repay-2-1-60days", _pr(2, 1, 60, "CRC")),
    ("h-repay-10-5", _pr(10, 5)),
]
 
REDTEAM = [  # (id, facts, bad output, expected code)
    ("promise-will-improve", _L, _out("Good news", _BASE_L + " These tips will improve your score.", _IMP), "PROMISE"),
    ("promise-guarantee", _L, _out("A plan", _BASE_L + " We guarantee a better score.", _IMP), BANNED_PHRASE),
    ("promise-approved", _L, _out("A plan", _BASE_L + " Then you could get approved for a loan.", _IMP), BANNED_PHRASE),
    ("promise-timeline", _L, _out("A plan", _BASE_L + " You can see results within a few weeks.", _IMP), "PROMISE"),
    ("number-invented", _D, _out("Your credit score dropped", _BASE_D.replace("15", "16"), _RPT), NUMBER_NOT_IN_FACTS),
    ("number-computed", _L, _out("Your credit score", _BASE_L + " You are 28 points away from the line.", _IMP), NUMBER_NOT_IN_FACTS),
    ("number-swapped", _D, _out("Your credit score dropped", "Your FirstCentral score moved from 695 to 710 between 14 Aug and 14 Sep, a drop of 15 points.", _RPT), NUMBER_ORDER),
    ("direction-rose", _D, _out("Your credit score changed", "Your FirstCentral score rose from 710 to 695 between 14 Aug and 14 Sep.", _RPT), WRONG_DIRECTION),
    ("points-wrong", _D, _out("Your credit score dropped", _BASE_D.replace("15 points", "710 points"), _RPT), POINTS_MISMATCH),
    ("missing-previous", _D, _out("Your credit score dropped", "Your FirstCentral score dropped to 695 on 14 Sep, a drop of 15 points.", _RPT), REQUIRED_FACT_MISSING),
    ("number-word", _P, _out("A note on your repayments", "We found two late or missed repayments in the last 90 days and 1 non-performing facility.", _REP), NUMBER_WORD),
    ("bureau-method", _L, _out("Your credit score", _BASE_L + " Scores are calculated from your payment history.", _IMP), "BUREAU_METHOD"),
    ("speculated-cause", _D, _out("Your credit score dropped", _BASE_D + " This was due to a recent loan application.", _RPT), "SPECULATED_CAUSE"),
    ("advice-invest", _L, _out("Your credit score", _BASE_L + " You could also consider investing.", _IMP), "ADVICE"),
    ("lender-named", _P, _out("A note on your repayments", _BASE_P + " Your GTBank loan is one of them.", _REP), LENDER),
    ("pii-bvn", _L, _out("Your credit score", _BASE_L + " Your BVN 12345678901 is on file.", _IMP), PII),
    ("pii-phone", _L, _out("Your credit score", _BASE_L + " We will call you on 08012345678.", _IMP), PII),
    ("pii-email", _L, _out("Your credit score", _BASE_L + " Write to help@example.com.", _IMP), PII),
    ("screen-not-allowed", _L, _out("Your credit score", _BASE_L, [("See loan offers", "loan_offers")]), SCREEN_NOT_ALLOWED),
    ("screen-wrong-trigger", _L, _out("Your credit score", _BASE_L, _REP), SCREEN_NOT_ALLOWED),
    ("format-not-json", _L, "Here is your message: " + _BASE_L, FORMAT_NOT_JSON),
    ("format-body-too-long", _L, _out("Your credit score", _BASE_L + " More detail. " * 40, _IMP), FORMAT_SCHEMA),
    ("format-title-too-long", _L, _out("A very long title that goes well past the sixty character limit set by the product requirements", _BASE_L, _IMP), FORMAT_SCHEMA),
    ("format-too-many-actions", _L, _out("Your credit score", _BASE_L, [("a", "improve_score")] * 4), FORMAT_SCHEMA),
    ("format-extra-field", _L, json.dumps({"title": "T", "body": _BASE_L, "actions": [], "why": "x"}), FORMAT_SCHEMA),
    ("empty-body", _L, json.dumps({"title": "T", "body": "   ", "actions": []}), EMPTY_TEXT),
    # ---- real failures from the first live Gemini run (they passed every older check) ----
    ("live-plural-1-points", _sd(650, 649, "2026-09-20", "2026-10-20"),
     _out("Your credit score dropped", "Your FirstCentral score moved from 650 to 649 between 20 Sep and 20 Oct, a drop of 1 points. Open your report to see what is on it.", _RPT), PLURAL),
    ("live-date-leading-zero", _sd(780, 702, "2026-07-05", "2026-10-05"),
     _out("Your credit score dropped", "Your FirstCentral score moved from 780 to 702 between 05 Jul and 05 Oct, a drop of 78 points. Open your report to see what is on it.", _RPT), DATE_FORMAT),
    ("live-late-repayments-on-report", _pr(2, 0),
     _out("A note on your repayments", "We found 2 late or missed repayments in the last 90 days on your FirstCentral report. Catching up on repayments is a good next step.", _REP), SOURCE_MISATTRIBUTION),
    ("live-late-repayments-on-report-registry", _pr(3, 0, 90, "CREDIT_REGISTRY"),
     _out("A note on your repayments", "We found 3 late or missed repayments in the last 90 days on your Credit Registry report. Catching up on your repayments is a good next step.", _REP), SOURCE_MISATTRIBUTION),
    ("source-credit-record", _pr(2, 0),
     _out("A note on your repayments", "We found 2 late or missed repayments in the last 90 days on your credit record. Catching up on repayments is a good next step.", _REP), SOURCE_MISATTRIBUTION),
    ("source-credit-file", _pr(3, 0),
     _out("A note on your repayments", "We found 3 late or missed repayments in the last 90 days in your credit file. Catching up on repayments is a good next step.", _REP), SOURCE_MISATTRIBUTION),
    ("plural-2-repayment", _pr(2, 0),
     _out("A note on your repayments", "We found 2 late or missed repayment in the last 90 days. Catching up is a good next step.", _REP), PLURAL),
]
 
 
def run_selftest() -> list[str]:
    fails: list[str] = []
 
    def expect(cond: bool, what: str):
        if not cond:
            fails.append(what)
 
    # -- guardrails: golden accepted, red-team rejected for the right reason, templates pass, examples pass
    for id_, facts, good in GOLDEN:
        r = check_output(good, parse_facts(facts))
        expect(r.ok, f"golden/{id_}: wrongly rejected {sorted(r.codes)}")
    for id_, facts, bad, code in REDTEAM:
        r = check_output(bad, parse_facts(facts))
        expect(not r.ok, f"redteam/{id_}: NOT rejected (hole in the checks)")
        expect(code in r.codes, f"redteam/{id_}: expected {code}, got {sorted(r.codes)}")
    rec = Recommender(ScriptedLLM([]))
    for id_, facts, _ in GOLDEN:
        f = parse_facts(facts)
        title, body, actions = rec._template_parts(f)
        r = check_output(json.dumps({"title": title, "body": body, "actions": [a.model_dump() for a in actions]}), f)
        expect(r.ok, f"template for golden/{id_} fails its own checks {sorted(r.codes)}")
    for code, (facts, answer) in PROMPT_EXAMPLES.items():
        r = check_output(json.dumps(answer), parse_facts({"trigger_code": code, **facts}))
        expect(r.ok, f"prompt example for {code} fails the checks {sorted(r.codes)}")
    # held-out facts: valid, templates work on them, and they really are NOT the prompt examples or golden facts
    seen = [parse_facts({"trigger_code": c, **f}).payload() for c, (f, _) in PROMPT_EXAMPLES.items()]
    seen += [parse_facts(f).payload() for _, f, _ in GOLDEN]
    for id_, facts in HELD_OUT:
        f = parse_facts(facts)
        expect(f.payload() not in seen, f"heldout/{id_}: duplicates a prompt example or golden case")
        title, body, actions = rec._template_parts(f)
        r = check_output(json.dumps({"title": title, "body": body, "actions": [a.model_dump() for a in actions]}), f)
        expect(r.ok, f"template for heldout/{id_} fails its own checks {sorted(r.codes)}")
 
    # -- the agent's behaviour
    def behaviour() -> None:
        good = GOLDEN[3][2]
        bad = _out("Good news", _BASE_L + " We guarantee it will improve.", [])
        low = _ls(572)
 
        llm = ScriptedLLM([good]); d = Recommender(llm).write(low)
        expect(d.origin is Origin.MODEL and d.attempts == 1 and d.publishable, "good first answer should be used, unflagged")
        expect(d.model_version == "scripted-1" and (d.input_tokens, d.output_tokens) == (100, 60), "versions/tokens recorded")
        expect(d.failed_attempt_codes == [], "a first-try success has no failed attempts")
 
        llm = ScriptedLLM([bad, good]); d = Recommender(llm).write(low)
        expect(d.origin is Origin.MODEL and d.attempts == 2, "retry should succeed")
        expect("previous answer was rejected" in llm.calls[1]["user"] and "guarantee" in llm.calls[1]["user"],
               "retry prompt must list the violations")
        expect("previous answer was rejected" not in llm.calls[0]["user"], "first prompt must not mention a rejection")
        expect(d.failed_attempt_codes == [["BANNED_PHRASE", "PROMISE"]] or
               (len(d.failed_attempt_codes) == 1 and {"BANNED_PHRASE", "PROMISE"} <= set(d.failed_attempt_codes[0])),
               "a retried draft must record WHY the first attempt failed")
 
        d = Recommender(ScriptedLLM([bad, "not json"])).write(low)
        expect(d.origin is Origin.TEMPLATE and d.needs_review and not d.publishable and d.attempts == 2,
               "two failures -> template + needs_review")
        expect([v.code for v in d.review_reasons] == [FORMAT_NOT_JSON], "review_reasons = the LAST attempt's violations")
        expect("572" in d.body and "600" in d.body, "template is filled from the facts")
 
        boom = RuntimeError("secret: BVN 12345678901")
        d = Recommender(ScriptedLLM([boom, boom])).write(low)
        expect(d.origin is Origin.TEMPLATE and d.needs_review and d.model_version is None, "provider errors -> template")
        expect([(v.code, v.detail) for v in d.review_reasons] == [("LLM_ERROR", "RuntimeError")], "error logged by TYPE only")
        expect(d.failed_attempt_codes == [["LLM_ERROR"], ["LLM_ERROR"]], "every failed attempt is recorded")
        expect("12345678901" not in d.model_dump_json(), "error message must not leak into the Draft")
 
        d = Recommender(ScriptedLLM([TimeoutError(), good])).write(low)
        expect(d.origin is Origin.MODEL and d.attempts == 2, "one error then success still works")
 
        llm = ScriptedLLM([good]); d = Recommender(llm).write(low)
        sent = json.loads(llm.calls[0]["user"].split("Facts (JSON): ")[1].splitlines()[0])
        expect(sent == d.trigger_facts, "trigger_facts must equal what was sent to the writer")
        blob = (llm.calls[0]["system"] + llm.calls[0]["user"]).lower()
        expect(not any(w in blob for w in ("subscriber_id", "full_name", "account_number", "phone_number", '"bvn"')),
               "no identifier field may reach the model")
        expect("improve_score, credit_report" in llm.calls[0]["system"], "screens in the prompt come from ALLOWED_SCREENS")
        expect(d.prompt_version == PROMPT_VERSION and d.prompt_hash == PROMPT_HASH, "prompt version/hash stamped")
 
        llm = ScriptedLLM([good])
        try:
            Recommender(llm).write({**low, "current_score": 700})
            expect(False, "invalid facts must raise")
        except ValidationError:
            expect(llm.calls == [], "the model must not be called for invalid facts")
 
        for extra in ("subscriber_id", "name", "bvn", "phone", "account_number"):
            try:
                parse_facts({**low, extra: "x"}); expect(False, f"contract must reject field {extra!r}")
            except ValidationError:
                pass
        for bad_facts, why in (({**_D, "points_dropped": 14}, "inconsistent points"),
                               ({**_D, "previous_snapshot_date": "2026-09-14", "current_snapshot_date": "2026-08-14"}, "dates out of order"),
                               ({**low, "current_score": 600}, "score not below the line"),
                               (_pr(0, 0), "no non-zero count"), ({"trigger_code": "CARD_USE", "bureau": "CRC"}, "unknown trigger")):
            try:
                parse_facts(bad_facts); expect(False, f"contract must reject: {why}")
            except ValidationError:
                pass
 
        llm = ScriptedLLM([good]); Recommender(llm, send_response_schema=True).write(low)
        expect(llm.calls[0]["json_schema"] and llm.calls[0]["json_schema"]["title"] == "Message", "schema sent when enabled")
        llm = ScriptedLLM([bad, bad, good]); d = Recommender(llm, max_attempts=3).write(low)
        expect(d.origin is Origin.MODEL and d.attempts == 3, "max_attempts is honoured")
 
 
    try:
        behaviour()
    except Exception as exc:  # a broken agent must show up as a failure, not as a crashed test
        fails.append(f"behaviour tests crashed: {type(exc).__name__}: {exc}")
 
    return fails
 
 
def _demo() -> None:
    facts = _sd(710, 695, "2026-08-14", "2026-09-14")
    bad = _out("Your credit score dropped", "Your FirstCentral score moved from 695 to 710 between 14 Aug and 14 Sep. We guarantee it will improve.", _RPT)
    llm = ScriptedLLM([bad, GOLDEN[0][2]])
    d = Recommender(llm).write(facts)
    print("--- prompt, attempt 1 (the part that changes per call) ---\n" + llm.calls[0]["user"][llm.calls[0]["user"].index("Trigger:"):])
    print("--- what the retry was told ---\n" + llm.calls[1]["user"].split("Fix them and answer again:\n")[1].split("\n")[0:3].__str__())
    print("--- Draft ---\n" + d.model_dump_json(indent=2))
 
 
def _example_reuse(draft: Draft) -> float | None:
    """0..1: how much the model's body repeats the prompt's worked example for this trigger, comparing
    only the sentences AFTER the first one, with digits masked. The first sentence is the reason and is
    dictated by the facts ("Your score is N, which is below 600"), so any honest writer repeats it. What
    follows (the advice and the closing) is where copying shows. Near 1.0 means the same sentences with
    different numbers. That is not automatically bad (consistent wording may be what the PM wants), but
    you should know it is happening: the model is filling in the example, not writing.
    None when it does not apply: the text came from the template, the facts ARE the example's facts
    (copying is then the right answer), or there is no second sentence to compare."""
    trig = draft.trigger_code.value
    ex_facts, ex_answer = PROMPT_EXAMPLES[trig]
    if draft.origin is not Origin.MODEL or draft.trigger_facts == ex_facts:
        return None
    def rest(text: str) -> str:  # everything after the first sentence, digits masked
        return re.sub(r"\d+", "#", " ".join(re.split(r"(?<=[.!?])\s+", text.strip())[1:]))
    mine, theirs = rest(draft.body), rest(ex_answer["body"])
    return difflib.SequenceMatcher(None, mine, theirs).ratio() if mine and theirs else None
 
 
REUSE_THRESHOLD = 0.90  # a starting guess for "same wording as the example"
 
 
def _run_live(llm: LLMClient, cases: list[tuple[str, dict]], *, runs: int = 1, show: bool = False,
              send_schema: bool = False) -> int:
    """Have a real model write a message for every case, `runs` times, and report how it went.
    Takes any LLMClient, so it can be tested with a stub."""
    rec = Recommender(llm, send_response_schema=send_schema)
    first = retried = fell = tin = tout = reused = 0
    sims: list[float] = []
    failure_codes: Counter = Counter()
    ever_failed: dict[str, int] = {}
    total = 0
 
    for run in range(1, runs + 1):
        if runs > 1:
            print(f"\n=== run {run}/{runs} ===")
        for id_, facts in cases:
            d = rec.write(facts)
            total += 1
            tin, tout = tin + d.input_tokens, tout + d.output_tokens
            sim = _example_reuse(d)
            if sim is not None:
                sims.append(sim)
                reused += sim >= REUSE_THRESHOLD
            for codes in d.failed_attempt_codes:
                failure_codes.update(codes)
            if d.failed_attempt_codes:
                ever_failed[id_] = ever_failed.get(id_, 0) + 1
            if d.origin is Origin.TEMPLATE:
                fell += 1; status = f"TEMPLATE  reasons={sorted({v.code for v in d.review_reasons})}"
            elif d.attempts == 1:
                first += 1; status = "ok first try"
            else:
                retried += 1; status = f"ok after retry  (first attempt rejected for {d.failed_attempt_codes[0]})"
            print(f"  {id_:<22} {status}")
            if show:
                print(f"      title  : {d.title}\n      body   : {d.body}")
                print("      actions: " + (" | ".join(f"{a.label} -> {a.screen}" for a in d.actions) or "(none)"))
                note = ("n/a" if sim is None else f"{sim:.2f}" + ("  <-- same wording as the prompt example" if sim >= REUSE_THRESHOLD else ""))
                print(f"      origin={d.origin.value}  attempts={d.attempts}  wording reuse of prompt example: {note}\n")
 
    print(f"\nmodel {llm.model_version}  prompt {PROMPT_VERSION}#{PROMPT_HASH}  schema sent: {send_schema}")
    print(f"{total} messages ({len(cases)} cases x {runs} run{'s' if runs != 1 else ''})")
    print(f"first try {first}/{total}  after retry {retried}/{total}  template fallback {fell}/{total}")
    print(f"tokens/msg {(tin + tout) / total:.0f}")
    if sims:
        print(f"wording reuse of the prompt examples (sentences after the first, digits masked, on facts unlike the example): "
              f"mean {sum(sims) / len(sims):.2f}, same wording (>= {REUSE_THRESHOLD}) in {reused}/{len(sims)}")
    else:
        print("wording reuse: no model-written messages to measure")
    if failure_codes:
        print("rejected attempts by reason: " + ", ".join(f"{c} x{n}" for c, n in failure_codes.most_common()))
        names = [f"{k} ({v}x)" for k, v in ever_failed.items()]
        print("cases that needed a retry at least once: " + ", ".join(names[:8]) + (f" ... and {len(names) - 8} more" if len(names) > 8 else ""))
    # The bar: template fallback should be rare (10% is a starting guess, not a tested threshold)
    return 1 if fell / total > 0.10 else 0
 
 
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="recommender.py", description="CreditCliq Recommendation Agent")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--selftest", action="store_true", help="guardrail + behaviour tests, no model (default)")
    mode.add_argument("--demo", action="store_true", help="show one retry with a scripted model")
    mode.add_argument("--schemas", action="store_true", help="print the input/output JSON contract")
    mode.add_argument("--live", action="store_true", help="have a real Gemini model write messages")
    ap.add_argument("--model", help="exact Gemini model id (for --live)")
    ap.add_argument("--show", action="store_true", help="(--live) print every message and its similarity to the prompt example")
    ap.add_argument("--runs", type=int, default=1, help="(--live) repeat the whole set N times")
    ap.add_argument("--set", choices=["golden", "heldout", "all"], default="all", dest="which",
                    help="(--live) which facts to use; heldout = facts the prompt never shows")
    ap.add_argument("--send-schema", action="store_true", help="(--live) also send the output JSON schema")
    args = ap.parse_args(argv)  # an unknown flag is an error, never silently ignored
 
    live_only = [f for f, v in (("--model", args.model), ("--show", args.show), ("--send-schema", args.send_schema))
                 if v] + (["--runs"] if args.runs != 1 else []) + (["--set"] if args.which != "all" else [])
    if live_only and not args.live:
        ap.error(f"{', '.join(live_only)} only apply to --live")
    if args.runs < 1:
        ap.error("--runs must be at least 1")
 
    if args.schemas:
        print(json.dumps({"input_TriggerFacts": _FACTS_ADAPTER.json_schema(), "output_Message": Message.model_json_schema()}, indent=2))
        return 0
    if args.demo:
        _demo(); return 0
 
    failures = run_selftest()  # always first: never test a model against broken guardrails
    if args.live:
        if not args.model:
            ap.error("--live needs --model <exact model id>")
        if failures:
            print("self-test failed; fix that first:", *failures, sep="\n - "); return 1
        cases = ([(i, f) for i, f, _ in GOLDEN] if args.which == "golden" else
                 list(HELD_OUT) if args.which == "heldout" else
                 [(i, f) for i, f, _ in GOLDEN] + list(HELD_OUT))
        return _run_live(GeminiClient(args.model), cases, runs=args.runs, show=args.show, send_schema=args.send_schema)
 
    print(f"self-test: {len(GOLDEN)} golden accepted, {len(REDTEAM)} red-team rejected, {len(HELD_OUT)} held-out facts and "
          f"templates verified, prompt examples verified")
    if failures:
        print("\nFAILED:", *failures, sep="\n - "); return 1
    print("ALL PASSED"); return 0
 
 
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))