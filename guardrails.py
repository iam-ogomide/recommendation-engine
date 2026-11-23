"""CreditCliq shared guardrail helpers, used by concierge.py and income.py.

Why this file exists
--------------------
concierge.py and income.py each carried their own copy of the same checks, and the copies had
drifted (three different lender lists) and shared three gaps:

  1. DATE DIGITS LEAKED INTO THE NUMBER CHECK. "2026-10-31" made 2026, 10 and 31 "allowed numbers",
     so an invented "dropped 10 points" or "DBR is 8" passed. Here dates are recognised as dates,
     checked against the dates in the facts, and REMOVED before the number check runs.
  2. WORDING RULES COULD NOT TELL A REFUSAL FROM A CLAIM. "I can't promise approval" was rejected
     for containing "approval". Here a clause that clearly REFUSES (can't promise / can't explain /
     can't predict ...) is exempt from the wording categories the caller names. Every other clause
     is still scanned, so "I can't promise approval, but you will qualify" is still caught.
  3. NO SHARED MODEL SEAM OR PII SHAPES. Those now come from recommender.py, as in guardian.py.

Plain code. No model, no network.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

from recommender import (  # shared with guardian.py and reviewer.py
    LENDER_NAMES as _BASE_LENDERS,
    Violation,
    numbers_in,
    _EMAIL, _LONG_DIGITS, _NG_PHONE, _NUMBER_WORDS,
)

__all__ = [
    "LENDER_NAMES", "LENDER_RE", "NUMBER_WORDS", "Violation", "numbers_in",
    "naira", "fmt_date", "collect_facts", "strip_dates", "mentions_date", "split_clauses",
    "scan_wording", "pii_hit", "has_pii_shape",
]

# One lender list for every agent. recommender.py's list plus the extra names income.py and
# concierge.py had added separately. Add new names HERE only.
_EXTRA_LENDERS = ("Carbon", "Branch", "Palmcredit", "Aella", "Lidya", "QuickCheck", "Sterling", "Keystone", "Providus")
LENDER_NAMES = tuple(dict.fromkeys(_BASE_LENDERS + _EXTRA_LENDERS))
LENDER_RE = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in LENDER_NAMES) + r")\b", re.IGNORECASE)
NUMBER_WORDS = _NUMBER_WORDS


# =============================================================================
# Formatting (what subscribers see)
# =============================================================================

def naira(n: int | float | None) -> str:
    """₦450,000. The number check already understands thousands separators."""
    if n is None:
        return "unknown"
    return f"₦{int(round(n)):,}"


def fmt_date(d: date | str) -> str:
    """31 Oct 2026 (no leading zero, same style as the Recommendation Agent)."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    return f"{d.day} {d.strftime('%b')} {d.year}"


# =============================================================================
# Dates: recognise them, compare them with the facts, take them out of the number check
# =============================================================================
_MON = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

_DATE_PATTERNS = (  # order matters: most specific first; matched spans are masked so they cannot overlap
    ("iso", re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")),
    ("dmy", re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + _MON + r")\b\.?(?:,?\s+(\d{4})\b)?", re.I)),
    ("mdy", re.compile(r"\b(" + _MON + r")\b\.?\s+(\d{1,2})(?!\d)(?:st|nd|rd|th)?(?:,?\s+(\d{4})\b)?", re.I)),
    ("my", re.compile(r"\b(" + _MON + r")\b\.?,?\s+(\d{4})\b", re.I)),
)

DateKey = tuple  # ("ymd", y, m, d) | ("md", m, d) | ("ym", y, m)


def _mon(s: str) -> int:
    return _MONTHS[s[:3].lower()]


def _key_from(kind: str, m: re.Match) -> DateKey | None:
    try:
        if kind == "iso":
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            date(y, mo, d)
            return ("ymd", y, mo, d)
        if kind == "dmy":
            d, mo, y = int(m.group(1)), _mon(m.group(2)), m.group(3)
            if y:
                date(int(y), mo, d)
                return ("ymd", int(y), mo, d)
            date(2000, mo, d)  # validity only (2000 is a leap year)
            return ("md", mo, d)
        if kind == "mdy":
            mo, d, y = _mon(m.group(1)), int(m.group(2)), m.group(3)
            if y:
                date(int(y), mo, d)
                return ("ymd", int(y), mo, d)
            date(2000, mo, d)
            return ("md", mo, d)
        if kind == "my":
            return ("ym", int(m.group(2)), _mon(m.group(1)))
    except ValueError:
        return None
    return None


def _scan_dates(text: str) -> list[tuple[int, int, DateKey]]:
    masked = text
    found: list[tuple[int, int, DateKey]] = []
    for kind, pat in _DATE_PATTERNS:
        for m in pat.finditer(masked):
            key = _key_from(kind, m)
            if key is None:
                continue
            found.append((m.start(), m.end(), key))
            masked = masked[:m.start()] + " " * (m.end() - m.start()) + masked[m.end():]
    return sorted(found)


def _known(key: DateKey, known: set[DateKey]) -> bool:
    if key in known:
        return True
    ymds = [k for k in known if k[0] == "ymd"]
    if key[0] == "md":
        return any((k[2], k[3]) == (key[1], key[2]) for k in ymds)
    if key[0] == "ym":
        return any((k[1], k[2]) == (key[1], key[2]) for k in ymds)
    return False


def _label(key: DateKey) -> str:
    if key[0] == "ymd":
        return f"{key[3]:02d}/{key[2]:02d}/{key[1]}"
    if key[0] == "md":
        return f"{key[2]:02d}/{key[1]:02d}"
    return f"{key[2]:02d}/{key[1]}"


def strip_dates(text: str, known: set[DateKey]) -> tuple[str, set[DateKey], list[str]]:
    """Remove every date from `text`. Returns (text without dates, known dates it mentioned,
    labels of dates that are NOT in the facts). A date the facts do not contain is a violation;
    a date the facts do contain is fine and must not feed the number check."""
    spans = _scan_dates(text)
    out, last = [], 0
    mentioned: set[DateKey] = set()
    unknown: list[str] = []
    for s, e, key in spans:
        out.append(text[last:s])
        out.append(" ")
        last = e
        if _known(key, known):
            mentioned.add(key)
        else:
            unknown.append(_label(key))
    out.append(text[last:])
    return "".join(out), mentioned, unknown


def mentions_date(mentioned: set[DateKey], d: date) -> bool:
    return ("ymd", d.year, d.month, d.day) in mentioned or ("md", d.month, d.day) in mentioned


def collect_facts(obj: Any) -> tuple[set[float], set[DateKey]]:
    """Walk tool or fact data. Returns (numbers that may appear, dates that may appear).
    Digits inside a date are NOT numbers: a string like '31 Oct 2026' contributes a date only."""
    nums: set[float] = set()
    dates: set[DateKey] = set()

    def walk(o: Any) -> None:
        if o is None or isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            nums.add(abs(float(o)))
        elif isinstance(o, str):
            for _, _, key in _scan_dates(o):
                dates.add(key)
            rest, _, _ = strip_dates(o, dates)
            nums.update(numbers_in(rest))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple, set)):
            for v in o:
                walk(v)

    walk(obj)
    return nums, dates


# =============================================================================
# Wording: scan clause by clause so a refusal is not punished for naming what it refuses
# =============================================================================
_CLAUSE_SPLIT = re.compile(r"[.!?;\n]+|,\s*(?:but|though|however|yet)\b|\b(?:but|however)\b", re.I)
_REFUSAL_FRAME = re.compile(
    r"\b(?:can(?:no|')t|cannot|can not|won'?t|will not|unable to|not able to|am not able to|do not|don'?t)\b"
    r"[^.]{0,30}?\b(?:promise|guarantee|predict|explain|say|tell|confirm|share|discuss|describe|calculate|"
    r"forecast|assure|know|speak to|comment)\b", re.I)


def split_clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT.split(text) if c and c.strip()]


def scan_wording(
    text: str,
    *,
    phrases: tuple[str, ...] = (),
    patterns: dict[str, tuple[str, ...]] | None = None,
    exempt: tuple[str, ...] = ("PROMISE", "BUREAU_METHOD", "BANNED_PHRASE"),
) -> list[tuple[str, str]]:
    """Return (code, matched text) for every wording rule that fires. A clause that clearly refuses
    ("I can't promise ...", "I can't explain ...") is skipped for the categories in `exempt` only."""
    hits: list[tuple[str, str]] = []
    for clause in split_clauses(text):
        refusing = bool(_REFUSAL_FRAME.search(clause))
        low = clause.lower()
        if not (refusing and "BANNED_PHRASE" in exempt):
            for ph in phrases:
                if ph.lower() in low:
                    hits.append(("BANNED_PHRASE", ph))
        for code, pats in (patterns or {}).items():
            if refusing and code in exempt:
                continue
            for p in pats:
                m = re.search(p, clause, re.I)
                if m:
                    hits.append((code, m.group(0)))
                    break
    return list(dict.fromkeys(hits))


# =============================================================================
# Identifier shapes
# =============================================================================

def has_pii_shape(text: str) -> bool:
    """Something that looks like a BVN, account number, phone number or email address."""
    return bool(_LONG_DIGITS.search(text) or _NG_PHONE.search(text) or _EMAIL.search(text))


def pii_hit(text: str) -> Violation | None:
    if has_pii_shape(text):
        return Violation(code="PII", detail="text contains something that looks like a BVN, account number, phone or email")
    return None