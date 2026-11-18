"""
MCP router: CreditChek identity, credit-bureau and Recovapro tools.

Mounted under /mcp by app/main.py. Two ways in:

  * in-process:  ``await execute_tool(name, args)``  (what enrich.py uses; no auth,
                 the caller is trusted code in the same app)
  * HTTP / SSE:  /mcp/call, /mcp/stream, /mcp/batch, /mcp/tools  (signed business
                 token required, per-business tool allow-list, rate limited)

Environment
-----------
CREDITCHEK_TOKEN              CreditChek API token (required for any lookup)
CREDITCHEK_BASE_URL           default https://api.creditchek.africa/v1
CREDITCHEK_VERIFY_TLS         default "true". Set "false" ONLY for a dev proxy.
BUSINESS_TOKEN_SECRET         HS256 secret for HTTP callers. If unset, every HTTP
                              route except /mcp/info answers 503 (fails closed).
MCP_MAX_BATCH                 max tools per /batch call            (default 10)
MCP_RATE_LIMIT_PER_MIN        calls per business per minute        (default 60)
MCP_CACHE_TTL_SECONDS         lookup cache TTL, 0 disables         (default 600)
MCP_CACHE_MAX_ENTRIES         cache size bound                     (default 1000)
MCP_MAX_CONCURRENCY           concurrent upstream calls per loop   (default 8)
MCP_VALIDATE_ARGS             "false" turns off schema checks      (default true)
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import copy
import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
import weakref
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/mcp", tags=["mcp"])


# ── CONFIG (read at call time so tests and hot config changes work) ──────────


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() not in {"0", "false", "no", ""}


def _base_url() -> str:
    return os.getenv("CREDITCHEK_BASE_URL", "https://api.creditchek.africa/v1").rstrip("/")


def _verify_tls() -> bool:
    return _env_bool("CREDITCHEK_VERIFY_TLS", True)


def _cache_ttl() -> int:
    return _env_int("MCP_CACHE_TTL_SECONDS", 600)


# ── LOG HYGIENE ──────────────────────────────────────────────────────────────


def mask(value: Any, keep: int = 2) -> str:
    """12345678901 -> 12*******01. Short values are fully masked."""
    s = str(value or "")
    if len(s) <= keep * 2:
        return "*" * len(s)
    return s[:keep] + "*" * (len(s) - keep * 2) + s[-keep:]


def _err(message: str, **extra: Any) -> dict[str, Any]:
    """The one error shape every handler returns.

    Callers should pass ``code`` (a stable machine-readable string) and
    ``retryable`` (True only when repeating the identical call could succeed
    and cannot double-bill). Clients branch on those, never on the message."""
    return {"error": True, "message": message, **extra}


def _http_failure(status: int) -> tuple[str, bool]:
    """HTTP status from CreditChek -> (code, retryable)."""
    if status == 429:
        return "rate_limited", True
    if status in (401,):
        return "auth", False
    if status == 403:
        return "forbidden", False
    if status in (400, 404):
        return "not_found", False
    if status >= 500:
        return "upstream_error", True
    return "upstream_error", False


# ── AUTHENTICATION & AUTHORISATION ───────────────────────────────────────────
class AuthError(Exception):
    pass


class BusinessContext(BaseModel):
    """Identity and permissions decoded from a signed business token."""

    business_id: str
    business_name: str = ""
    allowed_tools: list[str] = Field(default_factory=list)  # "*" means all
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    def can_call(self, tool_name: str) -> bool:
        return "*" in self.allowed_tools or tool_name in self.allowed_tools

    @property
    def is_admin(self) -> bool:
        return "*" in self.allowed_tools


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue_business_token(
    business_id: str,
    business_name: str,
    allowed_tools: list[str],
    ttl_seconds: int = 3600,
    secret: Optional[str] = None,
) -> str:
    """Mint an HS256 token. Run from an admin script, never from a request path."""
    secret = secret or os.getenv("BUSINESS_TOKEN_SECRET")
    if not secret:
        raise ValueError("BUSINESS_TOKEN_SECRET not set")
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url_encode(
        json.dumps(
            {
                "business_id": business_id,
                "business_name": business_name,
                "allowed_tools": allowed_tools,
                "exp": int(time.time()) + ttl_seconds,
            }
        ).encode()
    )
    sig = hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url_encode(sig)}"


def decode_business_token(token: str, secret: str) -> BusinessContext:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
        given_sig = _b64url_decode(sig_b64)
    except Exception:
        raise AuthError("malformed token")

    if header.get("alg") != "HS256":
        raise AuthError("unsupported token algorithm")

    expected = hmac.new(
        secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(expected, given_sig):
        raise AuthError("invalid token signature")

    exp = payload.get("exp")
    if not isinstance(exp, (int, float)):
        raise AuthError("token has no expiry")
    if exp < time.time():
        raise AuthError("token expired")
    if not payload.get("business_id"):
        raise AuthError("token has no business_id")

    allowed = payload.get("allowed_tools")
    return BusinessContext(
        business_id=str(payload["business_id"]),
        business_name=str(payload.get("business_name", "")),
        # A token that omits allowed_tools gets nothing, not everything.
        allowed_tools=[str(t) for t in allowed] if isinstance(allowed, list) else [],
        expires_at=datetime.fromtimestamp(exp, tz=timezone.utc),
    )


async def require_business(
    authorization: Optional[str] = Header(default=None),
) -> BusinessContext:
    secret = os.getenv("BUSINESS_TOKEN_SECRET")
    if not secret:
        raise HTTPException(503, "MCP authentication is not configured")
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            401, "Bearer token required", headers={"WWW-Authenticate": "Bearer"}
        )
    try:
        return decode_business_token(authorization[7:].strip(), secret)
    except AuthError as e:
        raise HTTPException(401, str(e), headers={"WWW-Authenticate": "Bearer"})


class _RateLimiter:
    """Sliding-window limiter, per business. In-process: with N workers the
    effective limit is N x the setting. Use Redis if that matters."""

    def __init__(self, window: float = 60.0):
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, cost: int = 1) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) + cost > limit:
                return False
            q.extend([now] * cost)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


_rate_limiter = _RateLimiter()


def _enforce_rate_limit(ctx: BusinessContext, cost: int = 1) -> None:
    limit = _env_int("MCP_RATE_LIMIT_PER_MIN", 60)
    if not _rate_limiter.allow(ctx.business_id, limit, cost):
        raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "30"})


def _enforce_tool_access(ctx: BusinessContext, tool_names: list[str]) -> None:
    unknown = [t for t in tool_names if t not in TOOL_INDEX]
    if unknown:
        raise HTTPException(404, f"Unknown tool: {unknown[0]}")
    denied = [t for t in tool_names if not ctx.can_call(t)]
    if denied:
        raise HTTPException(403, f"Token not permitted to call: {denied[0]}")


# ── METRICS ──────────────────────────────────────────────────────────────────


class _Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.calls: dict[str, int] = defaultdict(int)
        self.errors: dict[str, int] = defaultdict(int)
        self.total_ms: dict[str, float] = defaultdict(float)
        self.cache_hits = 0
        self.cache_misses = 0
        self.retries = 0

    def record(self, tool: str, ok: bool, ms: float) -> None:
        with self._lock:
            self.calls[tool] += 1
            self.total_ms[tool] += ms
            if not ok:
                self.errors[tool] += 1

    def bump(self, field: str) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + 1)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tools": {
                    t: {
                        "calls": n,
                        "errors": self.errors.get(t, 0),
                        "avg_ms": round(self.total_ms[t] / n, 1) if n else 0,
                    }
                    for t, n in self.calls.items()
                },
                "cache": {"hits": self.cache_hits, "misses": self.cache_misses},
                "upstream_retries": self.retries,
            }


metrics = _Metrics()


# ── PYDANTIC MODELS FOR HTTP API ─────────────────────────────────────────────


class ToolCallRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class ToolCallResponse(BaseModel):
    success: bool
    tool_name: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    retryable: Optional[bool] = None
    execution_time_ms: Optional[float] = None
    business_id: Optional[str] = None


class ServerInfo(BaseModel):
    name: str = "credit-intelligence-mcp-server"
    version: str = "2.1.0"
    protocol_version: str = "2025-06-18"
    capabilities: dict[str, Any] = Field(
        # The tool list is static for the life of the process, so listChanged is False.
        default_factory=lambda: {"tools": {"listChanged": False}, "streaming": True}
    )


# ── TOOL DEFINITIONS ─────────────────────────────────────────────────────────

BVN_PATTERN = r"^[0-9]{11}$"
DOB_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


def _s(description: Optional[str] = None, **extra: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"type": "string"}
    if description:
        d["description"] = description
    d.update(extra)
    return d


# MCP tool annotations: hints clients (and agent frameworks) use to decide what is
# safe to auto-approve or call in parallel. Lookups are read-only, repeatable and
# reach an external system.
_LOOKUP_HINTS = {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": True}


def _tool(
    name: str,
    description: str,
    properties: dict,
    required: list[str],
    annotations: Optional[dict[str, Any]] = None,
) -> dict:
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": properties, "required": required},
        "annotations": annotations or dict(_LOOKUP_HINTS),
    }


TOOLS: list[dict[str, Any]] = [
    # KYC
    _tool(
        "kyc_verify_bvn",
        "Verify BVN. If first_name/last_name/dob are supplied the result carries an "
        "identity_match block saying whether they agree with the BVN record.",
        {
            "bvn": _s("11-digit BVN", pattern=BVN_PATTERN),
            "first_name": _s("First name to match against the record"),
            "last_name": _s("Last name to match against the record"),
            "dob": _s("Date of birth (YYYY-MM-DD) to match", pattern=DOB_PATTERN),
        },
        ["bvn"],
    ),
    _tool(
        "kyc_verify_nin",
        "Verify NIN. Supplying first_name/last_name/dob adds an identity_match block.",
        {
            "nin": _s("11-digit NIN", pattern=BVN_PATTERN),
            "first_name": _s(),
            "last_name": _s(),
            "dob": _s("Date of birth (YYYY-MM-DD)", pattern=DOB_PATTERN),
        },
        ["nin"],
    ),
    _tool(
        "kyc_verify_account_number",
        "Verify bank account number and get account holder details",
        {
            "account_number": _s("10-digit account number"),  # no pattern: original did not enforce one
            "bank_code": _s("Bank code (e.g. 058 for GTBank)"),
        },
        ["account_number", "bank_code"],
    ),
    _tool(
        "kyc_verify_drivers_license",
        "Verify driver's license - REQUIRES first_name, last_name and dob",
        {
            "drivers_license": _s("Driver's license number"),
            "first_name": _s("First name (REQUIRED)"),
            "last_name": _s("Last name (REQUIRED)"),
            "dob": _s("Date of birth YYYY-MM-DD (REQUIRED)", pattern=DOB_PATTERN),
        },
        ["drivers_license", "first_name", "last_name", "dob"],
    ),
    _tool(
        "kyc_verify_international_passport",
        "Verify international passport - REQUIRES first_name, last_name and dob",
        {
            "passport_number": _s("Passport number"),
            "first_name": _s("First name (REQUIRED)"),
            "last_name": _s("Last name (REQUIRED)"),
            "dob": _s("Date of birth YYYY-MM-DD (REQUIRED)", pattern=DOB_PATTERN),
        },
        ["passport_number", "first_name", "last_name", "dob"],
    ),
    _tool(
        "kyc_verify_cac_rc",
        "Verify Corporate Affairs Commission (CAC) registration certificate",
        {
            "rc_number": _s("RC number (e.g. RC123456)"),
            "company_name": _s("Company name for matching"),
        },
        ["rc_number"],
    ),
    _tool(
        "kyc_intelligent_verify",
        "Verify a customer using the best identifier available "
        "(BVN > NIN > account > driver's license > passport > RC number)",
        {
            "customer_data": {
                "type": "object",
                "description": "Raw customer data with any available identifiers",
                "properties": {
                    k: _s()
                    for k in (
                        "bvn nin account_number bank_code drivers_license passport_number "
                        "rc_number company_name first_name last_name dob phone email"
                    ).split()
                },
            }
        },
        ["customer_data"],
    ),
    # Credit bureaus
    _tool(
        "credit_crc_individual",
        "Query CRC credit bureau for an individual report",
        {"bvn": _s(pattern=BVN_PATTERN)},
        ["bvn"],
    ),
    _tool(
        "credit_registry_individual",
        "Query Credit Registry for an individual",
        {"bvn": _s(pattern=BVN_PATTERN)},
        ["bvn"],
    ),
    _tool(
        "credit_firstcentral_individual",
        "Query FirstCentral credit bureau for an individual",
        {"bvn": _s(pattern=BVN_PATTERN)},
        ["bvn"],
    ),
    _tool(
        "credit_crc_sme",
        "Query CRC SME bureau for a business credit report",
        {"rc_number": _s("Company RC number")},
        ["rc_number"],
    ),
    _tool(
        "credit_firstcentral_sme",
        "Query FirstCentral SME bureau for a business",
        {"rc_number": _s("Company RC number")},
        ["rc_number"],
    ),
    _tool(
        "credit_sme_premium",
        "Query the CreditChek SME premium report for a business (higher cost)",
        {"rc_number": _s("Company RC number")},
        ["rc_number"],
    ),
    _tool(
        "credit_intelligent_bureau_check",
        "Query ALL relevant bureaus in parallel: the three individual bureaus when "
        "a bvn is given, CRC + FirstCentral SME when an rc_number is given",
        {
            "subject_data": {
                "type": "object",
                "description": "Raw subject data (individual and/or business)",
                "properties": {k: _s() for k in "bvn rc_number".split()},
            }
        },
        ["subject_data"],
    ),
    # Recovapro
    _tool(
        "recovapro_verify_mandate",
        "Verify e-mandate status via Recovapro (full result incl. borrower details)",
        {"mandate_id": _s("Mandate ID from Recovapro")},
        ["mandate_id"],
    ),
    _tool(
        "recovapro_get_mandate_details",
        "Get full mandate details (same payload as verify; kept for compatibility)",
        {"mandate_id": _s("Mandate ID")},
        ["mandate_id"],
    ),
    _tool(
        "recovapro_check_mandate_status",
        "Quick mandate status check: status, is_active, next debit date. No borrower PII.",
        {"mandate_id": _s("Mandate ID")},
        ["mandate_id"],
    ),
    # AML screening (PEP / sanctions / adverse media), served by the aml package
    _tool(
        "aml_screen_person",
        "Screen a person for PEP status, sanctions designations and adverse media in "
        "one call. Returns evidence with citations, not a lending decision. Writes an "
        "audit trace. Pass dob (YYYY-MM-DD) when known: it separates common names.",
        {
            "full_name": _s("Full legal name of the subject to screen"),
            "dob": _s("Date of birth (YYYY-MM-DD), if known", pattern=DOB_PATTERN),
            "business_id": _s("Business the subject is being screened for"),
        },
        ["full_name"],
        annotations={"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True},
    ),
    # recovapro_list_customer_mandates was removed: its schema required a field it
    # never accepted and no handler or CreditChek endpoint exists for it. Re-add it
    # once the real list endpoint is known.
]

TOOL_INDEX: dict[str, dict[str, Any]] = {t["name"]: t for t in TOOLS}


def _validate_args(tool_name: str, args: dict[str, Any]) -> list[str]:
    """Cheap pre-flight check so malformed input never costs a paid lookup.
    Messages never echo the offending value (it may be a BVN)."""
    schema = TOOL_INDEX[tool_name]["inputSchema"]
    problems = []
    for key in schema.get("required", []):
        if args.get(key) in (None, "", {}):
            problems.append(f"{key} is required")
    for key, spec in schema.get("properties", {}).items():
        value = args.get(key)
        pattern = spec.get("pattern")
        if value is None or not pattern:
            continue
        if not (isinstance(value, str) and re.search(pattern, value)):
            problems.append(f"{key} has an invalid format")
    return problems


# ── UPSTREAM TRANSPORT: shared client, retries, cache, single-flight ─────────


@dataclass
class UpstreamResponse:
    status_code: int
    body: Any  # parsed JSON, or None if the body was not JSON
    text: str  # first 500 chars of the raw body, for error messages


class _LoopState:
    """One httpx client / semaphore / in-flight map per event loop. The pipeline
    can call execute_tool from more than one loop (e.g. a worker thread), and an
    AsyncClient must not be shared across loops."""

    def __init__(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            verify=_verify_tls(),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
        self.sem = asyncio.Semaphore(_env_int("MCP_MAX_CONCURRENCY", 8))
        self.inflight: dict[str, asyncio.Future] = {}


try:
    _states: Any = weakref.WeakKeyDictionary()
except Exception:  # pragma: no cover
    _states = {}


def _state() -> _LoopState:
    loop = asyncio.get_running_loop()
    try:
        st = _states.get(loop)
        if st is None:
            st = _states[loop] = _LoopState()
    except TypeError:  # loop type not weak-referenceable
        key = id(loop)
        st = _states.setdefault(key, _LoopState()) if isinstance(_states, dict) else _LoopState()
    return st


async def aclose_http_client() -> None:
    """Call from the FastAPI lifespan shutdown hook."""
    try:
        st = _states.pop(asyncio.get_running_loop(), None)
    except Exception:
        st = None
    if st:
        await st.client.aclose()


class _TTLCache:
    def __init__(self) -> None:
        self._d: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            expires, value = item
            if expires < time.monotonic():
                del self._d[key]
                return None
            self._d.move_to_end(key)
            return copy.deepcopy(value)

    def set(self, key: str, value: Any, ttl: int) -> None:
        if ttl <= 0:
            return
        with self._lock:
            self._d[key] = (time.monotonic() + ttl, copy.deepcopy(value))
            self._d.move_to_end(key)
            while len(self._d) > _env_int("MCP_CACHE_MAX_ENTRIES", 1000):
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()


_cache = _TTLCache()

_RETRY_STATUS = {429}
_MAX_RETRIES = 2

# execute_tool(upstream_retries=N) overrides _MAX_RETRIES for the calls it makes.
# The in-process pipeline passes 0 and owns its own backoff, so a rate-limited
# lookup is retried by one layer, not two stacked ones.
_RETRIES: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "mcp_upstream_retries", default=None
)


# Whose lookup this is. The cache and single-flight are keyed on it so one business
# is never served another's result (each lender needs its own lookup, consent and
# audit trail). execute_tool sets it from its business_id; calls with none run in
# "_internal". MCP_CACHE_SCOPE=global turns the scoping off.
_CACHE_SCOPE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "mcp_cache_scope", default="_internal"
)


def _cache_scope() -> str:
    if os.getenv("MCP_CACHE_SCOPE", "business").strip().lower() == "global":
        return "_global"
    return _CACHE_SCOPE.get()


def _max_retries() -> int:
    override = _RETRIES.get()
    return _MAX_RETRIES if override is None else max(0, override)


def _cache_key(method: str, url: str, params: Any, json_body: Any) -> str:
    # Hashed so identifiers (BVN etc.) are not sitting in plain text as keys.
    raw = json.dumps(
        [_cache_scope(), method, url, params, json_body], sort_keys=True, default=str
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _cacheable(resp: UpstreamResponse) -> bool:
    if resp.status_code != 200 or resp.body is None:
        return False
    if isinstance(resp.body, dict):
        if resp.body.get("error") is True or resp.body.get("status") is False:
            return False
    return True


async def _send(
    method: str, url: str, params: Any, json_body: Any
) -> UpstreamResponse:
    """One logical request: concurrency-limited, retried on connect errors and
    HTTP 429 only. Read timeouts are NOT retried: the call may already have been
    processed (and billed)."""
    token = os.getenv("CREDITCHEK_TOKEN", "").strip()
    if not token:
        raise RuntimeError("CREDITCHEK_TOKEN not configured")
    headers = {
        "token": token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    st = _state()
    attempt = 0
    max_retries = _max_retries()
    while True:
        try:
            async with st.sem:
                resp = await st.client.request(
                    method, url, params=params, json=json_body, headers=headers
                )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            if attempt >= max_retries:
                raise
        else:
            if resp.status_code not in _RETRY_STATUS or attempt >= max_retries:
                try:
                    body = resp.json()
                except ValueError:
                    body = None
                return UpstreamResponse(resp.status_code, body, resp.text[:500])
        attempt += 1
        metrics.bump("retries")
        await asyncio.sleep(0.4 * (2 ** (attempt - 1)))


async def _creditchek_request(
    method: str,
    url: str,
    *,
    params: Optional[dict] = None,
    json_body: Optional[dict] = None,
    cache: bool = True,
) -> UpstreamResponse:
    """The single seam all handlers go through (tests monkeypatch this)."""
    ttl = _cache_ttl() if cache else 0
    if ttl <= 0:
        return await _send(method, url, params, json_body)

    key = _cache_key(method, url, params, json_body)
    hit = _cache.get(key)
    if hit is not None:
        metrics.bump("cache_hits")
        return hit
    metrics.bump("cache_misses")

    # Single-flight: identical concurrent lookups share one upstream call.
    st = _state()
    pending = st.inflight.get(key)
    if pending is not None:
        return copy.deepcopy(await asyncio.shield(pending))

    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    st.inflight[key] = fut
    try:
        resp = await _send(method, url, params, json_body)
        if _cacheable(resp):
            _cache.set(key, resp, ttl)
        fut.set_result(resp)
        return resp
    except asyncio.CancelledError:
        if not fut.done():
            fut.set_result(UpstreamResponse(0, {"error": True, "message": "cancelled"}, ""))
        raise
    except Exception as e:
        if not fut.done():
            fut.set_exception(e)
            fut.exception()  # mark retrieved so asyncio doesn't warn if nobody waits
        raise
    finally:
        st.inflight.pop(key, None)


def _transport_error(e: Exception) -> dict[str, Any]:
    if isinstance(e, httpx.TimeoutException):
        # Not retryable: the upstream may have processed (and billed) the call.
        return _err(
            "Request timeout - CreditChek took too long to respond",
            code="timeout", retryable=False,
        )
    if isinstance(e, httpx.ConnectError):
        return _err("Could not reach CreditChek", code="unreachable", retryable=True)
    return _err(str(e) or e.__class__.__name__, code="internal", retryable=False)


# ── IDENTITY MATCHING (BVN / NIN) ────────────────────────────────────────────

_FIRST_KEYS = ("firstname", "first_name", "givenname", "given_name")
_MIDDLE_KEYS = ("middlename", "middle_name", "othernames", "other_names")
_LAST_KEYS = ("lastname", "last_name", "surname", "familyname", "family_name")
_DOB_KEYS = ("dateofbirth", "date_of_birth", "dob", "birthdate", "birth_date")
_DOB_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%b-%Y", "%d %b %Y")


def _find_field(obj: Any, keys: tuple[str, ...], depth: int = 4) -> Optional[str]:
    """First string value stored under any of `keys` (case-insensitive), searched
    through nested dicts. CreditChek's exact field names are not documented in this
    repo, so the candidates above are tolerant guesses: verify against one real
    response and trim the lists."""
    if depth < 0 or not isinstance(obj, dict):
        return None
    for k, v in obj.items():
        if k.lower() in keys and isinstance(v, str) and v.strip():
            return v
    for v in obj.values():
        if isinstance(v, dict):
            found = _find_field(v, keys, depth - 1)
            if found:
                return found
    return None


def _name_tokens(s: str) -> set[str]:
    return set(re.sub(r"[^a-z\s-]", "", s.casefold()).replace("-", " ").split())


def _names_agree(given: str, record: str) -> bool:
    # Records often hold several given names; one shared token is a match.
    return bool(_name_tokens(given) & _name_tokens(record))


def _parse_dob(s: str) -> Optional[str]:
    s = s.strip().split("T")[0]
    for fmt in _DOB_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def build_identity_match(
    record: Any, first_name: Optional[str], last_name: Optional[str], dob: Optional[str]
) -> Optional[dict[str, Any]]:
    """Compare supplied applicant details with the verified record. Each field is
    True / False, or None when the record had nothing to compare against."""
    if not any([first_name, last_name, dob]):
        return None
    out: dict[str, Any] = {}
    if first_name:
        rec = _find_field(record, _FIRST_KEYS)
        # People often go by a middle name, and BVN records list it separately
        # (firstName JOSHUA, middleName IKECHUKWU): a given name matches either.
        middle = _find_field(record, _MIDDLE_KEYS)
        out["first_name"] = (
            None if rec is None else _names_agree(first_name, f"{rec} {middle or ''}")
        )
    if last_name:
        rec = _find_field(record, _LAST_KEYS)
        out["last_name"] = None if rec is None else _names_agree(last_name, rec)
    if dob:
        rec = _find_field(record, _DOB_KEYS)
        rec_dob = _parse_dob(rec) if rec else None
        given_dob = _parse_dob(dob)
        out["dob"] = None if (rec_dob is None or given_dob is None) else rec_dob == given_dob
    comparable = [v for v in out.values() if v is not None]
    out["checked"] = bool(comparable)
    out["matched"] = bool(comparable) and all(comparable)
    return out


# ── KYC HANDLER ──────────────────────────────────────────────────────────────


def _kyc_request(name: str, args: dict[str, Any]) -> tuple[str, Optional[dict], Optional[dict]]:
    """-> (url, query_params, json_body). Raises ValueError with a safe message."""
    base = _base_url()
    verify = f"{base}/identity/verifyData"

    def need(*keys: str) -> list[Any]:
        vals = [args.get(k) for k in keys]
        if not all(vals):
            raise ValueError(f"{', '.join(keys)} required")
        return vals

    if name == "kyc_verify_bvn":
        (bvn,) = need("bvn")
        logger.info("CreditChek BVN verification: %s", mask(bvn))
        return verify, {"bvn": bvn}, None
    if name == "kyc_verify_nin":
        (nin,) = need("nin")
        logger.info("CreditChek NIN verification: %s", mask(nin))
        return verify, {"nin": nin}, None
    if name == "kyc_verify_drivers_license":
        dl, first, last, dob = need("drivers_license", "first_name", "last_name", "dob")
        return verify, None, {
            "drivers_license": dl, "first_name": first, "last_name": last, "dob": dob,
        }
    if name == "kyc_verify_international_passport":
        pp, first, last, dob = need("passport_number", "first_name", "last_name", "dob")
        return verify, None, {
            "passport_number": pp, "first_name": first, "last_name": last, "dob": dob,
        }
    if name == "kyc_verify_cac_rc":
        (rc,) = need("rc_number")
        body = {"rc_number": rc}
        if args.get("company_name"):
            body["company_name"] = args["company_name"]
        return verify, None, body
    if name == "kyc_verify_account_number":
        acct, bank = need("account_number", "bank_code")
        logger.info("CreditChek account verification: %s @ bank %s", mask(acct), bank)
        return f"{base}/identity/account-verification", {
            "account_number": acct, "bank_code": bank,
        }, None
    raise ValueError(f"Unknown KYC tool: {name}")


async def _kyc_intelligent(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return _err("customer_data required")
    logger.info("Intelligent verify with keys: %s", sorted(data.keys()))
    who = {k: data.get(k) for k in ("first_name", "last_name", "dob")}
    has_who = all(who.values())

    if data.get("bvn"):
        return await handle_kyc_tool("kyc_verify_bvn", {"bvn": data["bvn"], **who})
    if data.get("nin"):
        return await handle_kyc_tool("kyc_verify_nin", {"nin": data["nin"], **who})
    if data.get("account_number") and data.get("bank_code"):
        return await handle_kyc_tool(
            "kyc_verify_account_number",
            {"account_number": data["account_number"], "bank_code": data["bank_code"]},
        )
    if data.get("drivers_license") and has_who:
        return await handle_kyc_tool(
            "kyc_verify_drivers_license", {"drivers_license": data["drivers_license"], **who}
        )
    if data.get("passport_number") and has_who:
        return await handle_kyc_tool(
            "kyc_verify_international_passport",
            {"passport_number": data["passport_number"], **who},
        )
    if data.get("rc_number"):
        sub = {"rc_number": data["rc_number"]}
        if data.get("company_name"):
            sub["company_name"] = data["company_name"]
        return await handle_kyc_tool("kyc_verify_cac_rc", sub)
    return _err("No valid identifier found in customer_data")


async def handle_kyc_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "kyc_intelligent_verify":
        return await _kyc_intelligent(args.get("customer_data") or {})
    try:
        url, params, body = _kyc_request(name, args)
    except ValueError as e:
        return _err(str(e), code="invalid_input", retryable=False)

    try:
        resp = await _creditchek_request("POST", url, params=params, json_body=body)
    except Exception as e:
        logger.error("KYC call failed for %s: %s", name, e)
        return _transport_error(e)

    result = resp.body
    if not isinstance(result, dict):
        logger.error("KYC %s: non-JSON/unexpected response (HTTP %s)", name, resp.status_code)
        code, retryable = _http_failure(resp.status_code)
        return _err(
            "Unexpected response format from CreditChek",
            status_code=resp.status_code, code=code, retryable=retryable,
        )
    if result.get("status") is True:
        out = dict(result)  # full CreditChek envelope, unchanged for existing callers
        if name in ("kyc_verify_bvn", "kyc_verify_nin"):
            match = build_identity_match(
                result.get("data", result),
                args.get("first_name"), args.get("last_name"), args.get("dob"),
            )
            if match is not None:
                out["identity_match"] = match
                if match["checked"] and not match["matched"]:
                    logger.warning("%s: supplied details do not match the record", name)
        return out
    logger.warning("KYC %s rejected: %s", name, result.get("message"))
    # The provider answered and said no (bad BVN, record not found...): final.
    return _err(
        result.get("message", "Unknown error from CreditChek"),
        details=result, code="rejected", retryable=False,
    )


# ── CREDIT BUREAU HANDLER ────────────────────────────────────────────────────

# tool -> (path under /credit, argument name, query-param name)
_CREDIT_ROUTES = {
    "credit_crc_individual": ("crc", "bvn", "bvn"),
    "credit_registry_individual": ("credit-registry", "bvn", "bvn"),
    "credit_firstcentral_individual": ("first-central", "bvn", "bvn"),
    "credit_crc_sme": ("sme/crc", "rc_number", "business_reg_no"),
    "credit_firstcentral_sme": ("sme/first-central", "rc_number", "business_reg_no"),
    "credit_sme_premium": ("sme/premium", "rc_number", "business_reg_no"),
}

_CREDIT_HTTP_ERRORS = {
    400: "Bad Request - the subject may not exist in this bureau's database",
    401: "Authentication failed - check CREDITCHEK_TOKEN",
    403: "Access forbidden - your API plan may not include this bureau",
}


async def _bureau_fanout(subject: dict[str, Any]) -> dict[str, Any]:
    """Query every relevant bureau concurrently. Individual bureaus when a BVN is
    present, SME bureaus when an RC number is present."""
    jobs: list[tuple[str, str, dict[str, Any]]] = []
    if bvn := subject.get("bvn"):
        jobs += [
            ("crc", "credit_crc_individual", {"bvn": bvn}),
            ("registry", "credit_registry_individual", {"bvn": bvn}),
            ("firstcentral", "credit_firstcentral_individual", {"bvn": bvn}),
        ]
    if rc := subject.get("rc_number"):
        jobs += [
            ("crc_sme", "credit_crc_sme", {"rc_number": rc}),
            ("firstcentral_sme", "credit_firstcentral_sme", {"rc_number": rc}),
        ]

    outcomes = await asyncio.gather(
        *(handle_credit_tool(tool, a) for _, tool, a in jobs), return_exceptions=True
    )
    results, failed = [], []
    for (label, _, _), out in zip(jobs, outcomes):
        if isinstance(out, BaseException):
            failed.append({"bureau": label, "error": str(out)})
        elif isinstance(out, dict) and out.get("error"):
            failed.append({
                "bureau": label,
                "error": out.get("message", "error"),
                "code": out.get("code"),
                "retryable": out.get("retryable"),
            })
        else:
            results.append({"bureau": label, "data": out})
    return {
        "bureaus_queried": len(results),   # successful bureaus (same meaning as before)
        "bureaus_failed": len(failed),
        "results": results,
        "failed": failed,
        "aggregated": True,
    }


async def handle_credit_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "credit_intelligent_bureau_check":
        return await _bureau_fanout(args.get("subject_data") or {})

    route = _CREDIT_ROUTES.get(name)
    if route is None:
        return _err(f"Unknown credit tool: {name}", code="unknown_tool", retryable=False)
    path, arg_name, param_name = route
    value = args.get(arg_name)
    if not value:
        return _err(f"{arg_name} required", code="invalid_input", retryable=False)

    url = f"{_base_url()}/credit/{path}"
    logger.info("Bureau lookup %s: %s", name, mask(value))
    try:
        resp = await _creditchek_request("GET", url, params={param_name: value})
    except Exception as e:
        logger.error("Bureau call failed for %s: %s", name, e)
        return _transport_error(e)

    if resp.status_code != 200:
        logger.error("Bureau %s HTTP %s: %s", name, resp.status_code, resp.text[:200])
        code, retryable = _http_failure(resp.status_code)
        return _err(
            _CREDIT_HTTP_ERRORS.get(resp.status_code, f"HTTP {resp.status_code}: {resp.text}"),
            status_code=resp.status_code,
            details=resp.text,
            code=code,
            retryable=retryable,
        )

    result = resp.body
    if result is None:
        return _err("Invalid JSON from API", raw_response=resp.text,
                    code="upstream_error", retryable=False)
    if not isinstance(result, dict):
        return _err(f"Unexpected response type: {type(result).__name__}",
                    raw_response=str(result)[:500], code="upstream_error", retryable=False)

    if result.get("error") is True:
        return _err(result.get("message", "Unknown error from bureau"),
                    details=result, code="rejected", retryable=False)
    if result.get("status") is True:
        return result.get("data", result)  # unwrapped, as before
    if "score" in result or "creditScore" in result or "data" in result:
        return result  # FirstCentral-style payload without the status wrapper
    logger.warning("Bureau %s: unknown response format, returning as-is", name)
    return result


# ── RECOVAPRO HANDLER ────────────────────────────────────────────────────────

_RECOVAPRO_TOOLS = {
    "recovapro_verify_mandate",
    "recovapro_get_mandate_details",
    "recovapro_check_mandate_status",
}


async def handle_recovapro_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name not in _RECOVAPRO_TOOLS:
        return _err(f"Unknown Recovapro tool: {name}", code="unknown_tool", retryable=False)
    mandate_id = arguments.get("mandate_id") or arguments.get("mandateId")
    if not mandate_id:
        return _err("mandate_id required", code="invalid_input", retryable=False)

    try:
        # Mandate state is live (balances, next debit): never cached.
        resp = await _creditchek_request(
            "POST",
            f"{_base_url()}/recova/consent/business/get-mandate",
            json_body={"mandateId": mandate_id},
            cache=False,
        )
    except Exception as e:
        return _transport_error(e)

    data = resp.body if isinstance(resp.body, dict) else None
    if resp.status_code != 200 or data is None:
        reason = (data or {}).get("message") if data else None
        code, retryable = _http_failure(resp.status_code)
        return _err(
            reason or "Mandate not found or access denied",
            status_code=resp.status_code,
            raw=data if data is not None else resp.text,
            code=code,
            retryable=retryable,
        )

    status = data.get("status")
    if name == "recovapro_check_mandate_status":
        return {
            "success": True,
            "mandate_id": mandate_id,
            "status": status,
            "is_active": status == "active",
            "next_debit_date": data.get("nextDebitDate"),
        }

    borrower = data.get("borrowerId", {}) or {}
    return {
        "success": True,
        "mandate_id": mandate_id,
        "status": status,
        "is_active": status == "active",
        "outstanding_balance": data.get("outstandingBalance", 0),
        "collected_amount": data.get("collectedAmount", 0),
        "total_amount_due": data.get("totalAmountDue", 0),
        "frequency": data.get("frequency"),
        "next_debit_date": data.get("nextDebitDate"),
        "borrower": {
            "fullname": borrower.get("fullname"),
            "bvn": borrower.get("bvn"),
            "phone": borrower.get("phone"),
            "email": borrower.get("email"),
        },
        "raw_response": data,
    }


# ── AML HANDLER ──────────────────────────────────────────────────────────────


async def handle_aml_tool(
    name: str, args: dict[str, Any], caller_business_id: Optional[str] = None
) -> dict[str, Any]:
    if name != "aml_screen_person":
        return _err(f"Unknown AML tool: {name}", code="unknown_tool", retryable=False)
    # Imported here so the credit/KYC tools still load where the aml package does not.
    from aml.service import run_aml_check

    # HTTP callers are scoped to their token's business: the audit trace must not
    # be writable under someone else's id. In-process callers (the pipeline) have
    # no token and pass business_id as an argument.
    business_id = caller_business_id or args.get("business_id")
    logger.info("AML screening requested (business=%s)", business_id)
    return await run_aml_check(
        full_name=args["full_name"],
        date_of_birth=args.get("dob"),
        business_id=business_id,
    )


# ── EXECUTION ────────────────────────────────────────────────────────────────


def _normalize(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Opt-in: give KYC results the same unwrapped shape credit tools use."""
    if (
        tool_name.startswith("kyc_")
        and isinstance(result, dict)
        and result.get("status") is True
        and isinstance(result.get("data"), dict)
    ):
        data = dict(result["data"])
        if "identity_match" in result:
            data["identity_match"] = result["identity_match"]
        return data
    return result


async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    normalize: bool = False,
    upstream_retries: Optional[int] = None,
    business_id: Optional[str] = None,
) -> dict[str, Any]:
    """Run a tool. Returns {success, tool_name, result, error, error_code,
    retryable, execution_time_ms}.

    normalize=True unwraps KYC responses to their `data` object (the shape the
    credit tools already return). Default False keeps today's shapes so existing
    callers keep working.

    upstream_retries overrides the transport-level retry count for this call
    (None = server default). Callers that run their own retry loop pass 0.
    business_id is the business the call is made for. HTTP callers pass their
    token's business: it overrides any business_id argument on tools that write
    per-business audit data, and it scopes the lookup cache (see _CACHE_SCOPE).
    """
    start = time.perf_counter()

    def finish(
        ok: bool,
        result: Any = None,
        error: Optional[str] = None,
        code: Optional[str] = None,
        retryable: bool = False,
    ) -> dict[str, Any]:
        ms = (time.perf_counter() - start) * 1000
        metrics.record(tool_name, ok, ms)
        return {
            "success": ok,
            "tool_name": tool_name,
            "result": result,
            "error": error,
            "error_code": code,
            "retryable": retryable,
            "execution_time_ms": ms,
        }

    if tool_name not in TOOL_INDEX:
        return finish(False, error=f"Unknown tool: {tool_name}", code="unknown_tool")

    arguments = arguments or {}
    if _env_bool("MCP_VALIDATE_ARGS", True):
        problems = _validate_args(tool_name, arguments)
        if problems:
            return finish(
                False, error="Invalid arguments: " + "; ".join(problems), code="invalid_input"
            )

    token = _RETRIES.set(upstream_retries) if upstream_retries is not None else None
    scope_token = _CACHE_SCOPE.set(str(business_id)) if business_id else None
    try:
        if tool_name.startswith("kyc_"):
            out = await handle_kyc_tool(tool_name, arguments)
        elif tool_name.startswith("credit_"):
            out = await handle_credit_tool(tool_name, arguments)
        elif tool_name.startswith("aml_"):
            out = await handle_aml_tool(tool_name, arguments, business_id)
        else:
            out = await handle_recovapro_tool(tool_name, arguments)
    except Exception as e:
        logger.exception("Error executing tool %s", tool_name)
        return finish(False, error=str(e), code="internal")
    finally:
        if token is not None:
            _RETRIES.reset(token)
        if scope_token is not None:
            _CACHE_SCOPE.reset(scope_token)

    if out.get("error"):
        return finish(
            False,
            error=out.get("message") or "Unknown error from API",
            code=out.get("code") or "upstream_error",
            retryable=bool(out.get("retryable", False)),
        )
    return finish(True, result=_normalize(tool_name, out) if normalize else out)


# ── HTTP / SSE ENDPOINTS ─────────────────────────────────────────────────────


def _to_response(
    tool_name: str, result: dict[str, Any], ctx: BusinessContext
) -> ToolCallResponse:
    return ToolCallResponse(
        success=result.get("success", False),
        tool_name=tool_name,
        result=result.get("result"),
        error=result.get("error"),
        error_code=result.get("error_code"),
        retryable=result.get("retryable"),
        execution_time_ms=result.get("execution_time_ms"),
        business_id=ctx.business_id,
    )


@router.get("/info")
async def get_server_info():
    return ServerInfo().model_dump()


@router.get("/tools")
async def list_tools(ctx: BusinessContext = Depends(require_business)):
    visible = [t for t in TOOLS if ctx.can_call(t["name"])]
    return {"tools": visible, "total": len(visible)}


@router.post("/call", response_model=ToolCallResponse)
async def call_tool(req: ToolCallRequest, ctx: BusinessContext = Depends(require_business)):
    _enforce_tool_access(ctx, [req.tool_name])
    _enforce_rate_limit(ctx)
    logger.info("Tool call %s by business %s", req.tool_name, ctx.business_id)
    result = await execute_tool(req.tool_name, req.arguments, business_id=ctx.business_id)
    return _to_response(req.tool_name, result, ctx)


@router.post("/stream")
async def stream_tool_call(
    request: ToolCallRequest, ctx: BusinessContext = Depends(require_business)
):
    # Checked before the stream opens so denials are real HTTP errors.
    _enforce_tool_access(ctx, [request.tool_name])
    _enforce_rate_limit(ctx)

    def sse(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"

    async def event_generator():
        try:
            yield sse("start", {"tool": request.tool_name, "timestamp": datetime.now(timezone.utc).isoformat()})
            result = await execute_tool(
                request.tool_name, request.arguments, business_id=ctx.business_id
            )
            yield sse("result", result)
            yield sse("complete", {"tool": request.tool_name, "timestamp": datetime.now(timezone.utc).isoformat()})
        except Exception as e:
            yield sse("error", {"error": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/batch")
async def batch_call_tools(
    requests: list[ToolCallRequest], ctx: BusinessContext = Depends(require_business)
):
    max_batch = _env_int("MCP_MAX_BATCH", 10)
    if len(requests) > max_batch:
        raise HTTPException(413, f"Batch too large: max {max_batch} calls")
    # All-or-nothing: refuse the whole batch before spending anything.
    _enforce_tool_access(ctx, [r.tool_name for r in requests])
    _enforce_rate_limit(ctx, cost=max(len(requests), 1))

    results = await asyncio.gather(
        *(execute_tool(r.tool_name, r.arguments, business_id=ctx.business_id) for r in requests)
    )  # order preserved; upstream concurrency is bounded by the semaphore
    return {
        "results": results,
        "total": len(results),
        "successful": sum(1 for r in results if r.get("success")),
        "business_id": ctx.business_id,
    }


@router.get("/metrics")
async def get_metrics(ctx: BusinessContext = Depends(require_business)):
    if not ctx.is_admin:
        raise HTTPException(403, "Admin token required")
    return metrics.snapshot()

# ── MCP PROTOCOL ENDPOINT (JSON-RPC 2.0 over Streamable HTTP, no sessions) ───
#
# POST /mcp is what a standard MCP client connects to (Claude, the Anthropic API
# `mcp_servers` connector, LangChain MCP adapters, IDEs). It reuses the same auth,
# allow-list, rate limit and execute_tool as the REST routes above, which stay for
# existing callers. Replies are plain application/json (the spec allows this); we
# do not open an SSE stream, so GET /mcp answers 405.

MCP_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

_MCP_INSTRUCTIONS = (
    "CreditChek identity, credit-bureau, e-mandate and AML tools for Nigerian lending. "
    "Bureau and KYC lookups are billed per call. Prefer recovapro_check_mandate_status "
    "over recovapro_verify_mandate unless borrower details are needed (it returns no PII). "
    "Pass first_name, last_name and dob to kyc_verify_bvn / kyc_verify_nin to get an "
    "identity_match block."
)


def _rpc_ok(id_: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _rpc_error(id_: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


@router.get("")
async def mcp_stream_not_offered():
    raise HTTPException(405, "This server does not offer an SSE stream", headers={"Allow": "POST"})


@router.post("")
async def mcp_rpc(request: Request, ctx: BusinessContext = Depends(require_business)):
    try:
        msg = await request.json()
    except Exception:
        return JSONResponse(_rpc_error(None, -32700, "Parse error"), status_code=400)
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        # Batches were dropped from the spec in 2025-06-18; not supported here.
        return JSONResponse(_rpc_error(None, -32600, "Invalid Request"), status_code=400)

    # Notifications (no id) and client responses need no reply body.
    if "id" not in msg or "method" not in msg:
        return Response(status_code=202)

    id_, method = msg["id"], msg["method"]
    params = msg.get("params") or {}
    if not isinstance(params, dict):
        return JSONResponse(_rpc_error(id_, -32602, "params must be an object"))

    if method == "initialize":
        wanted = params.get("protocolVersion")
        version = wanted if wanted in MCP_PROTOCOL_VERSIONS else MCP_PROTOCOL_VERSIONS[0]
        info = ServerInfo()
        return JSONResponse(_rpc_ok(id_, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": info.name, "version": info.version},
            "instructions": _MCP_INSTRUCTIONS,
        }))

    if method == "ping":
        return JSONResponse(_rpc_ok(id_, {}))

    if method == "tools/list":
        visible = [t for t in TOOLS if ctx.can_call(t["name"])]
        return JSONResponse(_rpc_ok(id_, {"tools": visible}))

    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(args, dict):
            return JSONResponse(_rpc_error(id_, -32602, "name and arguments are required"))
        # A tool the token cannot call is reported as unknown, matching tools/list.
        if name not in TOOL_INDEX or not ctx.can_call(name):
            return JSONResponse(_rpc_error(id_, -32602, f"Unknown tool: {name}"))
        _enforce_rate_limit(ctx)
        logger.info("MCP tools/call %s by business %s", name, ctx.business_id)
        res = await execute_tool(name, args, business_id=ctx.business_id)
        if res["success"]:
            payload = res["result"] if isinstance(res["result"], dict) else {"result": res["result"]}
        else:
            payload = {
                "error": res["error"],
                "code": res.get("error_code"),
                "retryable": res.get("retryable"),
            }
        return JSONResponse(_rpc_ok(id_, {
            "content": [{"type": "text", "text": json.dumps(payload, default=str)}],
            "structuredContent": payload,
            "isError": not res["success"],
        }))

    return JSONResponse(_rpc_error(id_, -32601, f"Method not found: {method}"))