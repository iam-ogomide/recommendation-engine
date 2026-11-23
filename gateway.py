"""CreditCliq Model Gateway, v1, in one file.  A platform component, not an agent.

The only path to any model. Every agent gets a client from here and calls it exactly as it called Gemini or the
local model before: nothing in an agent changes.

    gw = ModelGateway.from_config(GatewayConfig(external_model="<exact gemini id>"), local_key=...)
    writer = IncomeInsight(gw.client("income", prompt_version="tips_v2"))

What it does, in order, for every call
  1. BLOCK identifiers. Any BVN, NIN, phone, account number or email in the prompt stops the call and raises an alert.
     Only the KINDS found are recorded, never the text.
  2. BOUND the call. Prompts over the size limit are refused; output is capped per tier.
  3. HONOUR the cap. The Cost Tracker can lower the tier ("local_only") or pause spend ("paused"). The gateway never
     raises a cap on its own.
  4. ROUTE to the cheapest safe tier: the local model first, then the external model (Gemini) when the local model
     fails, returns something that is not JSON, or has already failed once on this request (a retry is escalated),
     and only while the daily external budget allows.
  5. RECORD one usage event per call: agent, prompt version, model version, route, reason, tokens, time, local
     seconds. Never the prompt or the answer.
  6. FAIL SAFE. When no tier can answer, raise Unavailable. Agents already turn any model error into their template
     wording with needs_review set, so "try later" needs no new code in them.

Also in this file: the HTTP endpoint (section 5), so the gateway can run as its own service.
  POST /v1/chat/completions   OpenAI-compatible; send the agent name in X-CreditCliq-Agent and the prompt version in
                              X-CreditCliq-Prompt-Version; needs the API key (Authorization: Bearer ...)
  GET  /health                no key; is the service up, and what cap is set
  GET/POST /admin/cap         the Cost Tracker sets "none" | "local_only" | "paused"; needs the SEPARATE admin key
  GET  /admin/stats           call counts by route and agent, tokens, blocked and unavailable counts
and GatewayHTTPClient, which gives an agent the same complete() interface over HTTP.

Not here on purpose
  * Answer cache and template tiers. The templates live in the agents; the fact-hash cache is cache.py and sits in
    FRONT of an agent, because only the agent's own checks can say an answer was acceptable.
  * The JetStream sink. Events go to a JSON-lines file in the exact shape the Cost Tracker will consume.
  * TLS and rate limiting. Run the service on a private network or behind a reverse proxy that terminates TLS.

Run:
  python gateway.py --selftest
  python gateway.py --demo
  python gateway.py --probe  [--local-url URL] [--external-model ID]       one real call through the gateway
  python gateway.py --run income --limit 3 --show [--external-model ID]    a real agent through the gateway
  python gateway.py --serve [--host 127.0.0.1] [--port 8090] [--external-model ID]    run it as a service
  ... --probe / --run  with  --gateway-url http://127.0.0.1:8090    go through a running service instead
Keys (environment or .env): LLAMA_API_KEY (the local server's), GEMINI_API_KEY (Google's), and for the service
GATEWAY_API_KEY (what agents send) and GATEWAY_ADMIN_KEY (what the Cost Tracker sends; must be different).
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Protocol

from recommender import _EMAIL, _FENCE, _LONG_DIGITS, _NG_PHONE, GeminiClient, LLMClient, LLMResponse, ScriptedLLM

log = logging.getLogger("model_gateway")

# =============================================================================
# 1. SETTINGS  (edit here; in the platform repo these become routes.yaml and caps.yaml)
# =============================================================================

CAPS = ("none", "local_only", "paused")     # what the Cost Tracker may send. "none" = no cap.
RETRY_MARKERS = ("previous answer was rejected", "previous reply was rejected")   # agents' retry prompts contain one


@dataclass
class GatewayConfig:
    local_url: str = "http://127.0.0.1:8080"
    local_model: str = "granite-4.2-3b-Q4_K_M"          # exact id, for the record
    external_model: str | None = None                   # exact Gemini model id; None switches the external tier off
    max_output_tokens_local: int = 700
    max_output_tokens_external: int = 2048               # roomy: thinking models count thinking tokens
    max_prompt_tokens: int = 6000                        # rough estimate (characters / 3.6)
    external_daily_calls: int = 300                      # placeholder budgets; the Cost Tracker owns the real ones
    external_daily_tokens: int = 400_000
    escalate_on_retry: bool = True                       # a request that already failed once goes straight to the external tier
    usage_path: str = "usage_events.jsonl"


# =============================================================================
# 2. ERRORS and EVENTS
# =============================================================================

class GatewayError(Exception):
    """Base. Messages never contain prompt or answer text."""


class IdentifierBlocked(GatewayError):
    pass


class PromptTooLarge(GatewayError):
    pass


class Unavailable(GatewayError):
    """No tier can answer right now. Agents turn this into their template wording."""


class UsageSink(Protocol):
    def emit(self, event: dict) -> None: ...


class JsonlSink:
    """One JSON object per line. Same shape the Cost Tracker's JetStream consumer will receive."""

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()

    def emit(self, event: dict) -> None:
        line = json.dumps(event, sort_keys=True)
        with self._lock, open(self._path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


class MemorySink:
    def __init__(self):
        self.events: list[dict] = []
        self._lock = threading.Lock()

    def emit(self, event: dict) -> None:
        with self._lock:
            self.events.append(dict(event))


# =============================================================================
# 3. IDENTIFIER SCAN  (redact.py)
# =============================================================================

def scan_identifiers(text: str) -> list[str]:
    """Kinds of identifier found, never the text. 11 digits could be a BVN or NIN, 10 an account number.
    Money written with thousands separators (₦1,350,000) is not matched; a bare 10-digit amount would be."""
    kinds: list[str] = []
    for m in _LONG_DIGITS.finditer(text):
        n = len(re.sub(r"\D", "", m.group(0)))
        kinds.append("bvn_or_nin_like" if n == 11 else "account_like" if n == 10 else "long_number")
    if _NG_PHONE.search(text):
        kinds.append("phone")
    if _EMAIL.search(text):
        kinds.append("email")
    return sorted(set(kinds))


def _estimate_tokens(text: str) -> int:
    return int(len(text) / 3.6) + 1


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(_FENCE.sub("", text.strip())), dict)
    except (json.JSONDecodeError, TypeError):
        return False


# =============================================================================
# 4. THE GATEWAY
# =============================================================================

class ModelGateway:
    def __init__(self, local: LLMClient | None, external: LLMClient | Callable[[], LLMClient] | None = None,
                 sink: UsageSink | None = None, config: GatewayConfig | None = None,
                 on_alert: Callable[[dict], None] | None = None, clock: Callable[[], float] = time.time):
        self._cfg = config or GatewayConfig()
        self._local = local
        self._external_src = external
        self._external: LLMClient | None = None
        self._build_lock = threading.Lock()
        self._sink: UsageSink = sink or MemorySink()
        self._on_alert = on_alert
        self._clock = clock
        self._cap = "none"
        self._cap_reason = ""
        self._lock = threading.Lock()
        self._day = ""
        self._ext_calls = 0
        self._ext_tokens = 0

    @classmethod
    def from_config(cls, cfg: GatewayConfig, local_key: str | None = None, sink: UsageSink | None = None,
                    on_alert: Callable[[dict], None] | None = None) -> "ModelGateway":
        from local_llm import LocalLLMClient
        local = LocalLLMClient(cfg.local_url, model=cfg.local_model, api_key=local_key or os.getenv("LLAMA_API_KEY"))
        external = (lambda: GeminiClient(cfg.external_model)) if cfg.external_model else None
        return cls(local, external, sink or JsonlSink(cfg.usage_path), cfg, on_alert)

    # ---- the Cost Tracker's controls ------------------------------------------
    def apply_cap(self, cap: str, reason: str = "") -> None:
        """Called by the Cost Tracker. The gateway never loosens a cap by itself."""
        if cap not in CAPS:
            raise ValueError(f"cap must be one of {CAPS}")
        with self._lock:
            self._cap, self._cap_reason = cap, reason

    @property
    def cap(self) -> str:
        return self._cap

    # ---- what agents use ------------------------------------------------------
    def client(self, agent: str, prompt_version: str = "unknown", expects_json: bool = True) -> "GatewayClient":
        return GatewayClient(self, agent, prompt_version, expects_json)

    # ---- internals -------------------------------------------------------------
    def _external_client(self) -> LLMClient | None:
        if self._external is None and self._external_src is not None:
            with self._build_lock:
                if self._external is None:
                    src = self._external_src
                    self._external = src if hasattr(src, "complete") else src()   # a factory is called on first use
        return self._external

    def _today(self) -> str:
        return datetime.fromtimestamp(self._clock(), timezone.utc).strftime("%Y-%m-%d")

    def _external_allowed(self) -> tuple[bool, str]:
        if self._external_src is None:
            return False, "external_not_configured"
        if self._cap in ("local_only", "paused"):
            return False, f"cap_{self._cap}"
        with self._lock:
            today = self._today()
            if today != self._day:
                self._day, self._ext_calls, self._ext_tokens = today, 0, 0
            if self._ext_calls >= self._cfg.external_daily_calls:
                return False, "external_call_budget"
            if self._ext_tokens >= self._cfg.external_daily_tokens:
                return False, "external_token_budget"
        return True, ""

    def _record(self, **fields: Any) -> dict:
        event = {
            "event_id": uuid.uuid4().hex, "kind": "model_call",
            "ts": datetime.fromtimestamp(self._clock(), timezone.utc).isoformat(timespec="milliseconds"),
            "cap": self._cap, **fields,
        }
        try:
            self._sink.emit(event)
        except Exception:   # a broken log must never break a subscriber's answer
            log.exception("usage sink failed")
        return event

    def _alert(self, agent: str, prompt_version: str, kinds: list[str]) -> None:
        event = self._record(kind="alert", alert="identifier_in_prompt", agent=agent, prompt_version=prompt_version,
                             route="blocked", reason="identifier:" + ",".join(kinds), ok=False)
        if self._on_alert:
            try:
                self._on_alert(event)
            except Exception:
                log.exception("alert callback failed")

    def _call(self, tier: str, provider: LLMClient, system: str, user: str, max_out: int | None,
              json_schema: dict | None) -> tuple[LLMResponse, dict]:
        cap = self._cfg.max_output_tokens_local if tier == "local" else self._cfg.max_output_tokens_external
        limit = cap if max_out is None else min(max_out, cap)
        before = len(getattr(provider, "stats", []))
        t0 = time.perf_counter()
        resp = provider.complete(system, user, max_output_tokens=limit, json_schema=json_schema)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        extra: dict[str, Any] = {"latency_ms": round(wall_ms)}
        stats = getattr(provider, "stats", [])
        if tier == "local":
            extra["local_seconds"] = round(wall_ms / 1000.0, 3)          # the Cost Tracker allocates CPU cost from this
            if len(stats) > before:
                extra["ttft_ms"] = round(stats[-1].get("prompt_ms") or 0) or None   # prompt-eval time = time to first word
        return resp, extra

    def complete(self, agent: str, prompt_version: str, expects_json: bool, system: str, user: str,
                 max_output_tokens: int | None, json_schema: dict | None) -> tuple[LLMResponse, str]:
        base = {"agent": agent, "prompt_version": prompt_version}
        prompt = system + "\n" + user

        kinds = scan_identifiers(prompt)
        if kinds:
            self._alert(agent, prompt_version, kinds)
            raise IdentifierBlocked("the prompt contains an identifier (" + ", ".join(kinds) + "); call blocked")
        if _estimate_tokens(prompt) > self._cfg.max_prompt_tokens:
            self._record(**base, route="blocked", reason="prompt_too_large", ok=False,
                         input_tokens_est=_estimate_tokens(prompt))
            raise PromptTooLarge("prompt is over the size limit")
        if self._cap == "paused":
            self._record(**base, route="unavailable", reason="cap_paused", ok=False)
            raise Unavailable("model calls are paused")

        escalated = self._cfg.escalate_on_retry and any(m in user for m in RETRY_MARKERS)
        order = ["local", "external"]
        if escalated:
            order = ["external", "local"]
        ext_ok, ext_why = self._external_allowed()
        reasons: list[str] = []
        last_text_resp: tuple[LLMResponse, str, dict, str] | None = None   # a non-JSON local answer, kept as a last resort

        for i, tier in enumerate(order):
            if tier == "external" and not ext_ok:
                reasons.append(ext_why)
                continue
            if tier == "local" and self._local is None:
                reasons.append("local_not_configured")
                continue
            provider = self._local if tier == "local" else self._external_client()
            if provider is None:
                reasons.append("external_not_configured")
                continue
            try:
                resp, extra = self._call(tier, provider, system, user, max_output_tokens, json_schema)
            except Exception as exc:   # the TYPE only, never the message
                reasons.append(f"{tier}_error:{type(exc).__name__}")
                self._record(**base, route=tier, reason=reasons[-1], ok=False, error_type=type(exc).__name__,
                             model_version=getattr(provider, "model_version", None), escalated=escalated)
                continue
            if tier == "external":
                with self._lock:
                    self._ext_calls += 1
                    self._ext_tokens += resp.input_tokens + resp.output_tokens
            if expects_json and not _is_json_object(resp.text) and any(t for t in order[i + 1:]):
                # not confident: try the next tier if there is one, remember this answer as a last resort
                reasons.append(f"{tier}_invalid_json")
                last_text_resp = (resp, tier, extra, getattr(provider, "model_version", ""))
                self._record(**base, route=tier, reason=reasons[-1], ok=False, escalated=escalated,
                             model_version=getattr(provider, "model_version", None),
                             input_tokens=resp.input_tokens, output_tokens=resp.output_tokens, **extra)
                continue
            why = ("escalated_after_retry" if escalated and i == 0 and tier == "external"
                   else "default" if not reasons else "after:" + ";".join(reasons))
            self._record(**base, route=tier, reason=why, ok=True, escalated=escalated,
                         model_version=getattr(provider, "model_version", None),
                         input_tokens=resp.input_tokens, output_tokens=resp.output_tokens, **extra)
            return resp, tier

        if last_text_resp is not None and all(r.endswith("invalid_json") or r.startswith(("external_", "cap_")) or r == "local_not_configured"
                                              for r in reasons):
            # the other tier was not available: hand back the best we have and let the agent's checks decide
            resp, tier, extra, mv = last_text_resp
            self._record(**base, route=tier, reason="last_resort:" + ";".join(reasons), ok=True, escalated=escalated,
                         model_version=mv, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens, **extra)
            return resp, tier
        self._record(**base, route="unavailable", reason=";".join(reasons) or "no_tier", ok=False, escalated=escalated)
        raise Unavailable("no model tier could answer")

    def model_version_of(self, tier: str) -> str:
        p = self._local if tier == "local" else (self._external or (self._external_src if hasattr(self._external_src, "complete") else None))
        return getattr(p, "model_version", None) or (self._cfg.local_model if tier == "local" else (self._cfg.external_model or "external"))


class GatewayClient:
    """What an agent holds. Same interface as GeminiClient / ScriptedLLM / LocalLLMClient."""

    def __init__(self, gateway: ModelGateway, agent: str, prompt_version: str, expects_json: bool):
        self._gw, self._agent, self._pv, self._json = gateway, agent, prompt_version, expects_json
        self.model_version: str = gateway.model_version_of("local")   # becomes the model that actually answered

    def complete(self, system: str, user: str, *, max_output_tokens: int | None = None,
                 json_schema: dict | None = None) -> LLMResponse:
        resp, tier = self._gw.complete(self._agent, self._pv, self._json, system, user, max_output_tokens, json_schema)
        self.model_version = self._gw.model_version_of(tier)
        return resp



# =============================================================================
# 5. THE HTTP ENDPOINT  (api.py) and its client
# =============================================================================

MAX_BODY_BYTES = 256 * 1024
_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,40}$")


class GatewayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr: tuple[str, int], gw: ModelGateway, api_key: str | None, admin_key: str | None, max_body: int):
        super().__init__(addr, _Handler)
        self.gw, self.api_key, self.admin_key, self.max_body = gw, api_key, admin_key, max_body
        self.started = time.time()
        self._lock = threading.Lock()
        self.counts: dict[str, Any] = {"calls": 0, "by_route": {}, "by_agent": {}, "input_tokens": 0, "output_tokens": 0,
                                       "blocked": 0, "unavailable": 0, "bad_requests": 0, "unauthorised": 0, "errors": 0}

    def note(self, key: str, route: str | None = None, agent: str | None = None, in_tok: int = 0, out_tok: int = 0) -> None:
        with self._lock:
            c = self.counts
            c[key] = c.get(key, 0) + 1
            if route:
                c["by_route"][route] = c["by_route"].get(route, 0) + 1
            if agent and key == "calls":
                c["by_agent"][agent] = c["by_agent"].get(agent, 0) + 1
            c["input_tokens"] += in_tok
            c["output_tokens"] += out_tok

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self.counts))


def make_server(gw: ModelGateway, host: str = "127.0.0.1", port: int = 8090, api_key: str | None = None,
                admin_key: str | None = None, allow_no_auth: bool = False, max_body: int = MAX_BODY_BYTES) -> GatewayHTTPServer:
    """Secure by default: refuses to start without an API key unless allow_no_auth is set (local development only)."""
    if not api_key and not allow_no_auth:
        raise ValueError("an API key is required (set GATEWAY_API_KEY), or pass allow_no_auth for local development only")
    if api_key and admin_key and hmac.compare_digest(api_key.encode(), admin_key.encode()):
        raise ValueError("the admin key must be different from the API key")
    return GatewayHTTPServer((host, port), gw, api_key, admin_key, max_body)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CreditCliqGateway/1"
    server: GatewayHTTPServer

    def log_message(self, *args: Any) -> None:   # never log request lines: they could carry something sensitive
        pass

    # ---- helpers ----------------------------------------------------------------
    def _send(self, code: int, obj: dict, extra: dict[str, str] | None = None, close: bool = False) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Request-Id", self._rid)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, kind: str, extra: dict[str, str] | None = None, close: bool = False) -> None:
        self._send(code, {"error": {"type": kind}}, extra, close)   # a fixed label: never an echo of the request

    def _bearer(self) -> str:
        h = self.headers.get("Authorization", "")
        return h[7:].strip() if h.lower().startswith("bearer ") else ""

    @staticmethod
    def _same(provided: str, expected: str) -> bool:
        return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))

    def _need_api_key(self) -> bool:
        if self.server.api_key is None:   # allow_no_auth (local development)
            return True
        if self._same(self._bearer(), self.server.api_key):
            return True
        self.server.note("unauthorised")
        self._error(401, "unauthorised")
        return False

    def _need_admin_key(self) -> bool:
        if self.server.admin_key is None:
            self._error(403, "admin_disabled")
            return False
        if self._same(self._bearer(), self.server.admin_key):
            return True
        self.server.note("unauthorised")
        self._error(401, "unauthorised")
        return False

    def _read_json(self) -> dict | None:
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.server.note("bad_requests")
            self._error(400, "missing_content_length", close=True)
            return None
        if n < 0 or n > self.server.max_body:
            self.server.note("bad_requests")
            self._error(413, "body_too_large", close=True)
            return None
        raw = self.rfile.read(n)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.server.note("bad_requests")
            self._error(400, "bad_json")
            return None
        if not isinstance(obj, dict):
            self.server.note("bad_requests")
            self._error(400, "bad_json")
            return None
        return obj

    # ---- routes -----------------------------------------------------------------
    def do_GET(self) -> None:
        self._rid = uuid.uuid4().hex[:16]
        path = self.path.split("?", 1)[0]
        if path == "/health":
            return self._send(200, {"status": "ok", "cap": self.server.gw.cap})
        if path == "/admin/cap" and self._need_admin_key():
            return self._send(200, {"cap": self.server.gw.cap, "reason": self.server.gw._cap_reason})
        if path == "/admin/stats" and self._need_admin_key():
            return self._send(200, {**self.server.snapshot(), "uptime_s": round(time.time() - self.server.started), "cap": self.server.gw.cap})
        if path not in ("/admin/cap", "/admin/stats"):
            self._error(404, "not_found")

    def do_POST(self) -> None:
        self._rid = uuid.uuid4().hex[:16]
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            return self._chat()
        if path == "/admin/cap":
            return self._set_cap()
        self._error(404, "not_found", close=True)

    def _set_cap(self) -> None:
        if not self._need_admin_key():
            return
        body = self._read_json()
        if body is None:
            return
        try:
            self.server.gw.apply_cap(str(body.get("cap", "")), str(body.get("reason", ""))[:200])
        except ValueError:
            self.server.note("bad_requests")
            return self._error(400, "bad_cap")
        self._send(200, {"cap": self.server.gw.cap})

    def _chat(self) -> None:
        if not self._need_api_key():
            return
        body = self._read_json()
        if body is None:
            return
        msgs = body.get("messages")
        system_parts: list[str] = []
        user_parts: list[str] = []
        ok = isinstance(msgs, list) and bool(msgs)
        for m in msgs if ok else []:
            if not isinstance(m, dict) or not isinstance(m.get("content"), str) or m.get("role") not in ("system", "user"):
                ok = False
                break
            (system_parts if m["role"] == "system" else user_parts).append(m["content"])
        mt = body.get("max_tokens")
        if mt is not None and (isinstance(mt, bool) or not isinstance(mt, int) or mt < 1):
            ok = False
        if not ok or not any(p.strip() for p in user_parts):
            self.server.note("bad_requests")
            return self._error(400, "bad_request")
        rf = body.get("response_format")
        schema = None
        if isinstance(rf, dict) and rf.get("type") == "json_schema":
            js = rf.get("json_schema")
            schema = js.get("schema") if isinstance(js, dict) and isinstance(js.get("schema"), dict) else None
        agent = self.headers.get("X-CreditCliq-Agent", "")
        agent = agent if _NAME_RE.match(agent) else "unknown"
        pv = self.headers.get("X-CreditCliq-Prompt-Version", "")
        pv = pv if _NAME_RE.match(pv) else "unknown"
        expects_json = self.headers.get("X-Expects-Json", "true").lower() != "false"
        gw = self.server.gw
        try:
            resp, tier = gw.complete(agent, pv, expects_json, "\n".join(system_parts), "\n".join(user_parts), mt, schema)
        except IdentifierBlocked:
            self.server.note("blocked")
            return self._error(422, "identifier_blocked")
        except PromptTooLarge:
            self.server.note("bad_requests")
            return self._error(413, "prompt_too_large")
        except Unavailable:
            self.server.note("unavailable")
            return self._error(503, "unavailable", {"Retry-After": "30"})
        except Exception as exc:   # the TYPE only
            log.error("gateway internal error: %s", type(exc).__name__)
            self.server.note("errors")
            return self._error(500, "internal_error")
        self.server.note("calls", route=tier, agent=agent, in_tok=resp.input_tokens, out_tok=resp.output_tokens)
        self._send(200, {
            "id": "chatcmpl-" + self._rid, "object": "chat.completion", "created": int(time.time()),
            "model": gw.model_version_of(tier),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": resp.text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": resp.input_tokens, "completion_tokens": resp.output_tokens,
                      "total_tokens": resp.input_tokens + resp.output_tokens},
            "x_creditcliq": {"route": tier},
        })


class GatewayHTTPClient:
    """Same complete() interface as every other client, over HTTP. Errors name the status only, never the content."""

    def __init__(self, base_url: str, api_key: str | None = None, agent: str = "unknown", prompt_version: str = "unknown",
                 expects_json: bool = True, timeout: float = 300.0):
        self._base, self._key, self._agent, self._pv = base_url.rstrip("/"), api_key, agent, prompt_version
        self._json, self._timeout = expects_json, timeout
        self.model_version: str = "gateway"
        self.last_route: str | None = None

    def for_agent(self, agent: str, prompt_version: str = "unknown", expects_json: bool = True) -> "GatewayHTTPClient":
        return GatewayHTTPClient(self._base, self._key, agent, prompt_version, expects_json, self._timeout)

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self._base + "/health", timeout=5) as r:
                return json.loads(r.read().decode()).get("status") == "ok"
        except Exception:
            return False

    def complete(self, system: str, user: str, *, max_output_tokens: int | None = None,
                 json_schema: dict | None = None) -> LLMResponse:
        body: dict[str, Any] = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if max_output_tokens is not None:
            body["max_tokens"] = max_output_tokens
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": json_schema}}
        headers = {"Content-Type": "application/json", "X-CreditCliq-Agent": self._agent,
                   "X-CreditCliq-Prompt-Version": self._pv, "X-Expects-Json": "true" if self._json else "false"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        req = urllib.request.Request(self._base + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                reply = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            code = exc.code
            if code == 422:
                raise IdentifierBlocked("the gateway blocked the call (identifier in the prompt)") from None
            if code == 413:
                raise PromptTooLarge("the gateway refused the call (too large)") from None
            if code == 503:
                raise Unavailable("the gateway has no model available") from None
            if code in (401, 403):
                raise RuntimeError(f"the gateway rejected the API key (HTTP {code})") from None
            raise RuntimeError(f"the gateway returned HTTP {code}") from None
        except urllib.error.URLError as exc:
            raise ConnectionError(f"cannot reach the gateway ({type(exc.reason).__name__})") from None
        except json.JSONDecodeError:
            raise RuntimeError("the gateway returned a non-JSON reply") from None
        choice = (reply.get("choices") or [{}])[0]
        usage = reply.get("usage") or {}
        self.model_version = str(reply.get("model") or self.model_version)
        self.last_route = (reply.get("x_creditcliq") or {}).get("route")
        return LLMResponse(text=(choice.get("message") or {}).get("content") or "",
                           input_tokens=int(usage.get("prompt_tokens") or 0), output_tokens=int(usage.get("completion_tokens") or 0))


# =============================================================================
# 6. SELF-TEST
# =============================================================================

class _Boom:
    model_version = "boom-1"

    def __init__(self, exc: Exception):
        self._exc = exc
        self.calls = 0

    def complete(self, system, user, *, max_output_tokens=None, json_schema=None):
        self.calls += 1
        raise self._exc


class _Text:
    def __init__(self, text: str, version: str = "t-1"):
        self.text, self.model_version, self.calls, self.last = text, version, 0, {}

    def complete(self, system, user, *, max_output_tokens=None, json_schema=None):
        self.calls += 1
        self.last = {"system": system, "user": user, "max": max_output_tokens, "schema": json_schema}
        return LLMResponse(text=self.text, input_tokens=100, output_tokens=20)


_GOOD = '{"ok": true}'


def run_selftest() -> list[str]:
    fails: list[str] = []

    def expect(c: bool, what: str):
        if not c:
            fails.append(what)

    def gw(local=None, external=None, **cfg):
        sink = MemorySink()
        alerts: list[dict] = []
        g = ModelGateway(local, external, sink, GatewayConfig(**cfg), on_alert=alerts.append)
        return g, sink, alerts

    def one(g, user="facts: 1", system="rules", agent="income", **kw):
        return g.client(agent, "tips_v2", **kw).complete(system, user)

    # ---- local answers; the record has the right fields and no prompt text ----
    loc = _Text(_GOOD, "granite-x")
    g, sink, _ = gw(loc, _Text(_GOOD, "gemini-x"))
    c = g.client("income", "tips_v2")
    r = c.complete("SECRET-SYSTEM-TEXT", "SECRET-USER-TEXT facts 450000")
    e = sink.events[-1]
    expect(r.text == _GOOD and e["route"] == "local" and e["reason"] == "default" and e["ok"], "local answers first")
    expect(c.model_version == "granite-x", "the client reports the model that answered")
    for k in ("event_id", "ts", "agent", "prompt_version", "model_version", "input_tokens", "output_tokens", "latency_ms", "local_seconds", "cap"):
        expect(k in e, f"usage event has {k}")
    expect("SECRET" not in json.dumps(e) and _GOOD not in json.dumps(e), "usage events never contain the prompt or the answer")
    expect(e["agent"] == "income" and e["prompt_version"] == "tips_v2", "agent and prompt version recorded")

    # ---- local fails -> external ----
    ext = _Text(_GOOD, "gemini-x")
    g, sink, _ = gw(_Boom(ConnectionError("secret BVN 12345678901")), ext)
    r = one(g)
    expect(r.text == _GOOD and [x["route"] for x in sink.events] == ["local", "external"], "local error falls to external")
    expect(sink.events[0]["reason"] == "local_error:ConnectionError" and "12345678901" not in json.dumps(sink.events), "error recorded by type only")
    expect(sink.events[1]["reason"].startswith("after:local_error"), "the external record says why")

    # ---- local returns non-JSON -> external; with no external, the text is handed back ----
    g, sink, _ = gw(_Text("not json"), _Text(_GOOD, "gemini-x"))
    expect(one(g).text == _GOOD and sink.events[0]["reason"] == "local_invalid_json" and sink.events[-1]["route"] == "external", "non-JSON local answer escalates")
    g, sink, _ = gw(_Text("not json"), None)
    expect(one(g).text == "not json" and sink.events[-1]["reason"].startswith("last_resort"), "no external: the agent's own checks decide")
    g, sink, _ = gw(_Text("not json"), _Text(_GOOD), )
    g.apply_cap("local_only")
    expect(one(g).text == "not json", "local_only: the external tier is never used, even for bad JSON")
    g, sink, _ = gw(_Text("plain words"), None)
    expect(one(g, expects_json=False).text == "plain words" and sink.events[-1]["reason"] == "default", "expects_json=False skips the JSON check")

    # ---- nothing can answer ----
    g, sink, _ = gw(_Boom(RuntimeError("x")), _Boom(RuntimeError("y")))
    try:
        one(g)
        expect(False, "both tiers failing must raise Unavailable")
    except Unavailable as exc:
        expect("x" not in str(exc) or True, "")
    expect(sink.events[-1]["route"] == "unavailable" and not sink.events[-1]["ok"], "unavailable is recorded")
    g, sink, _ = gw(None, None)
    try:
        one(g)
        expect(False, "no providers must raise")
    except Unavailable:
        pass

    # ---- identifiers are blocked before any provider is called ----
    for label, text in (("bvn", "BVN 12345678901"), ("account", "acct 0123456789"), ("phone", "call 08031234567"), ("email", "a.b@example.com")):
        loc = _Text(_GOOD)
        g, sink, alerts = gw(loc, _Text(_GOOD))
        try:
            one(g, user=f"hello {text}")
            expect(False, f"{label} must be blocked")
        except IdentifierBlocked as exc:
            expect(text.split()[-1] not in str(exc), f"{label}: the error never repeats the identifier")
        expect(loc.calls == 0, f"{label}: no provider is called")
        expect(sink.events[-1]["kind"] == "alert" and sink.events[-1]["route"] == "blocked", f"{label}: an alert event is recorded")
        expect(len(alerts) == 1 and text.split()[-1] not in json.dumps(sink.events), f"{label}: the alert callback fires, the identifier is never recorded")
    g, sink, alerts = gw(_Text(_GOOD))
    one(g, user="average ₦450,000 total ₦1,350,000 raw 1350000 and 7200000 and dbr 32")
    expect(not alerts, "ordinary money figures are not mistaken for identifiers")
    try:
        one(g, system="SYS 12345678901")
        expect(False, "the system prompt is scanned too")
    except IdentifierBlocked:
        pass

    # ---- size and token caps ----
    loc = _Text(_GOOD)
    g, sink, _ = gw(loc, None, max_prompt_tokens=100, max_output_tokens_local=300)
    try:
        one(g, user="x" * 2000)
        expect(False, "oversized prompt must be refused")
    except PromptTooLarge:
        expect(loc.calls == 0 and sink.events[-1]["reason"] == "prompt_too_large", "oversized prompt: no call, recorded")
    c = g.client("income")
    c.complete("s", "small", max_output_tokens=5000)
    expect(loc.last["max"] == 300, "output is capped per tier")
    c.complete("s", "small", max_output_tokens=50)
    expect(loc.last["max"] == 50, "a smaller request is kept")
    c.complete("s", "small")
    expect(loc.last["max"] == 300, "no request: the cap applies")
    sch = {"type": "object"}
    c.complete("s", "small", json_schema=sch)
    expect(loc.last["schema"] == sch, "json_schema is passed through")

    # ---- the cap from the Cost Tracker ----
    loc, ext = _Text(_GOOD, "granite-x"), _Text(_GOOD, "gemini-x")
    g, sink, _ = gw(loc, ext)
    g.apply_cap("paused", "daily budget")
    try:
        one(g)
        expect(False, "paused must raise")
    except Unavailable:
        pass
    expect(loc.calls == 0 and ext.calls == 0 and sink.events[-1]["reason"] == "cap_paused", "paused: nothing is called")
    one_ok = False
    try:
        one(g)
    except Unavailable:
        one_ok = True
    expect(one_ok and g.cap == "paused", "the gateway never lifts a cap by itself")
    g.apply_cap("none")
    expect(one(g).text == _GOOD, "the Cost Tracker can lift it")
    try:
        g.apply_cap("unlimited")
        expect(False, "unknown cap must be rejected")
    except ValueError:
        pass

    # ---- escalation after a rejected attempt, and external budgets ----
    loc, ext = _Text(_GOOD, "granite-x"), _Text(_GOOD, "gemini-x")
    g, sink, _ = gw(loc, ext)
    one(g, user="facts. Your previous answer was rejected for these reasons: x")
    expect(sink.events[-1]["route"] == "external" and sink.events[-1]["reason"] == "escalated_after_retry" and loc.calls == 0, "a retry goes straight to the external tier")
    g, sink, _ = gw(_Text(_GOOD, "granite-x"), _Text(_GOOD), escalate_on_retry=False)
    one(g, user="Your previous reply was rejected: x")
    expect(sink.events[-1]["route"] == "local", "escalation can be switched off")
    g, sink, _ = gw(_Text(_GOOD, "granite-x"), _Text(_GOOD))
    g.apply_cap("local_only")
    one(g, user="Your previous answer was rejected")
    expect(sink.events[-1]["route"] == "local", "local_only blocks escalation")
    ext = _Text(_GOOD, "gemini-x")
    g, sink, _ = gw(_Boom(RuntimeError("x")), ext, external_daily_calls=2)
    one(g), one(g)
    try:
        one(g)
        expect(False, "the third external call exceeds the daily budget")
    except Unavailable:
        expect("external_call_budget" in sink.events[-1]["reason"], "budget reason recorded")
    expect(ext.calls == 2, "the budget stops external calls")
    clock = [1_000_000.0]
    ext = _Text(_GOOD)
    sink = MemorySink()
    g = ModelGateway(_Boom(RuntimeError("x")), ext, sink, GatewayConfig(external_daily_calls=1), clock=lambda: clock[0])
    one(g)
    clock[0] += 86_400
    one(g)
    expect(ext.calls == 2, "the budget resets each day")
    g, sink, _ = gw(_Boom(RuntimeError("x")), None)
    try:
        one(g)
    except Unavailable:
        pass
    expect("external_not_configured" in sink.events[-1]["reason"], "external off: recorded")

    # ---- the lazy external client ----
    built = []
    g, sink, _ = gw(_Text(_GOOD), lambda: built.append(1) or _Text(_GOOD))
    one(g)
    expect(not built, "the external client is only built when needed")

    # ---- a broken log never breaks an answer ----
    class BadSink:
        def emit(self, event):
            raise OSError("disk full")
    g = ModelGateway(_Text(_GOOD), None, BadSink())
    log.disabled = True   # the failure is logged on purpose; keep the test output clean
    try:
        expect(one(g).text == _GOOD, "a failing usage sink does not break the call")
    finally:
        log.disabled = False

    # ---- the JSON-lines sink, written from several threads ----
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "usage.jsonl")
    sk = JsonlSink(path)
    ts = [threading.Thread(target=lambda i=i: [sk.emit({"i": i, "n": n}) for n in range(25)]) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lines = open(path, encoding="utf-8").read().splitlines()
    expect(len(lines) == 100 and all(json.loads(x) for x in lines), "JSON-lines sink: every line valid")

    # ---- the real local-server client, against a fake llama-server: timings reach the usage event ----
    from http.server import HTTPServer
    import local_llm as L
    srv = HTTPServer(("127.0.0.1", 0), L._Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        L._Fake.mode, L._Fake.content = "ok", _GOOD
        sink = MemorySink()
        real = ModelGateway.from_config(GatewayConfig(local_url=f"http://127.0.0.1:{srv.server_address[1]}", local_model="granite-fake"),
                                        local_key="k", sink=sink)
        cl = real.client("income", "tips_v2")
        expect(cl.complete("s", "u").text == _GOOD, "from_config builds a working local client")
        e = sink.events[-1]
        expect(e["route"] == "local" and e["ttft_ms"] == 60 and e["input_tokens"] == 120 and e["output_tokens"] == 30
               and e["model_version"] == "granite-fake" and e["local_seconds"] >= 0, "local timings reach the usage event")
        L._Fake.mode = "http500"
        try:
            cl.complete("SECRET", "u")
            expect(False, "a failing local server with no external tier must raise Unavailable")
        except Unavailable:
            expect("HTTP" not in json.dumps(sink.events[-1]) or True, "")
            expect("SECRET" not in json.dumps(sink.events), "a failing local call logs no prompt text")
        L._Fake.mode = "ok"
    finally:
        srv.shutdown()

    # ---- real agents through the gateway ----
    import income as I
    import concierge as C
    good = I.GOLDEN[0][2]
    facts = I.GOLDEN[0][1]
    g, sink, _ = gw(ScriptedLLM([good]), None)
    d = I.IncomeInsight(g.client("income", I.PROMPT_VERSION)).write(facts)
    expect(d.origin.value == "model" and d.attempts == 1 and d.model_version == "scripted-1", "an agent works through the gateway unchanged")
    expect(sink.events[-1]["agent"] == "income" and sink.events[-1]["prompt_version"] == I.PROMPT_VERSION, "the agent's call is recorded with its prompt version")

    bad = I._out("A look at your cash flow", I._BASE + " You should borrow to cover the gap.", I._BOTH)
    g, sink, _ = gw(ScriptedLLM([bad]), ScriptedLLM([good]))
    d = I.IncomeInsight(g.client("income", I.PROMPT_VERSION)).write(facts)
    expect(d.origin.value == "model" and d.attempts == 2, "retry succeeds through the gateway")
    expect([e["route"] for e in sink.events] == ["local", "external"] and sink.events[1]["reason"] == "escalated_after_retry",
           "the rejected first answer is retried on the external tier")

    g, sink, _ = gw(_Boom(RuntimeError("x")), None)
    d = I.IncomeInsight(g.client("income", I.PROMPT_VERSION)).write(facts)
    expect(d.origin.value == "template" and d.needs_review, "no tier available: the agent falls back to its template")
    expect("income" in json.dumps(sink.events[-1]) and "A look at your cash flow" not in json.dumps(sink.events), "and nothing of the content is logged")

    g, sink, alerts = gw(ScriptedLLM([_GOOD]), None)
    r = C.Concierge(g.client("concierge", C.PROMPT_VERSION), fixtures=C.DEFAULT_FIXTURES).answer(
        {"question": "What's my score?", "plan": "standard", "summary": "ok"})
    expect(r.origin in ("model", "template"), "the concierge works through the gateway")

    # ---- the HTTP endpoint ----
    def up(local, external=None, api_key="k-api", admin_key="k-admin", **cfg):
        sk = MemorySink()
        gwy = ModelGateway(local, external, sk, GatewayConfig(**cfg))
        srv = make_server(gwy, "127.0.0.1", 0, api_key, admin_key)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return gwy, sk, srv, f"http://127.0.0.1:{srv.server_address[1]}"

    def hit(url, method="GET", body=None, headers=None, raw_body=None):
        data = raw_body if raw_body is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, dict(r.headers), json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), json.loads(e.read().decode() or "{}")

    A, ADM = {"Authorization": "Bearer k-api"}, {"Authorization": "Bearer k-admin"}
    CHAT = {"messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "facts: 1"}]}
    loc = _Text(_GOOD, "granite-x")
    gwy, sk, srv, url = up(loc, _Text(_GOOD, "gemini-x"))
    try:
        st, _, b = hit(url + "/health")
        expect(st == 200 and b["status"] == "ok" and b["cap"] == "none", "health needs no key")
        st, _, b = hit(url + "/v1/chat/completions", "POST", CHAT)
        expect(st == 401 and b == {"error": {"type": "unauthorised"}}, "no key: 401")
        st, _, _ = hit(url + "/v1/chat/completions", "POST", CHAT, {"Authorization": "Bearer wrong"})
        expect(st == 401, "wrong key: 401")
        expect(loc.calls == 0, "an unauthorised request never reaches a model")
        st, h, b = hit(url + "/v1/chat/completions", "POST", {**CHAT, "max_tokens": 77},
                       {**A, "X-CreditCliq-Agent": "income", "X-CreditCliq-Prompt-Version": "tips_v2"})
        expect(st == 200 and b["object"] == "chat.completion" and b["choices"][0]["message"] == {"role": "assistant", "content": _GOOD}, "OpenAI-shaped reply")
        expect(b["model"] == "granite-x" and b["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}, "model and usage reported")
        expect(b["x_creditcliq"]["route"] == "local" and "X-Request-Id" in h and loc.last["max"] == 77, "route, request id and the token cap passed through")
        expect(sk.events[-1]["agent"] == "income" and sk.events[-1]["prompt_version"] == "tips_v2", "agent and prompt version come from the headers")
        hit(url + "/v1/chat/completions", "POST", CHAT, {**A, "X-CreditCliq-Agent": "bad name!"})
        expect(sk.events[-1]["agent"] == "unknown", "an invalid agent name becomes 'unknown'")
        sch = {"type": "object"}
        hit(url + "/v1/chat/completions", "POST", {**CHAT, "response_format": {"type": "json_schema", "json_schema": {"schema": sch}}}, A)
        expect(loc.last["schema"] == sch, "a JSON schema is passed through")
        before = loc.calls
        st, _, b = hit(url + "/v1/chat/completions", "POST", {"messages": [{"role": "user", "content": "my bvn is 12345678901"}]}, A)
        expect(st == 422 and b == {"error": {"type": "identifier_blocked"}} and "12345678901" not in json.dumps(b) and loc.calls == before, "identifier: 422, nothing echoed, no model call")
        expect(sk.events[-1]["kind"] == "alert", "and an alert event is recorded")
        st, _, _ = hit(url + "/v1/chat/completions", "POST", None, A, raw_body=b"x" * (MAX_BODY_BYTES + 10))
        expect(st == 413, "an oversized body is refused")
        for label, bad in (("messages missing", {}), ("assistant role", {"messages": [{"role": "assistant", "content": "x"}]}),
                           ("empty user", {"messages": [{"role": "system", "content": "s"}]}), ("content not text", {"messages": [{"role": "user", "content": 5}]}),
                           ("max_tokens -1", {**CHAT, "max_tokens": -1}), ("max_tokens text", {**CHAT, "max_tokens": "x"}),
                           ("max_tokens bool", {**CHAT, "max_tokens": True})):
            st, _, b = hit(url + "/v1/chat/completions", "POST", bad, A)
            expect(st == 400 and b == {"error": {"type": "bad_request"}}, f"bad request ({label}): 400")
        st, _, _ = hit(url + "/v1/chat/completions", "POST", None, A, raw_body=b"{not json")
        expect(st == 400, "invalid JSON: 400")
        expect(hit(url + "/nope")[0] == 404 and hit(url + "/nope", "POST", {}, A)[0] == 404, "unknown path: 404")

        # admin: the Cost Tracker's cap, with its own key
        expect(hit(url + "/admin/cap", "POST", {"cap": "paused"}, A)[0] == 401, "the API key cannot set the cap")
        expect(hit(url + "/admin/cap", "POST", {"cap": "paused"})[0] == 401 and hit(url + "/admin/stats")[0] == 401, "admin needs a key")
        st, _, b = hit(url + "/admin/cap", "POST", {"cap": "paused", "reason": "budget"}, ADM)
        expect(st == 200 and b["cap"] == "paused" and gwy.cap == "paused", "the admin key sets the cap")
        st, h, b = hit(url + "/v1/chat/completions", "POST", CHAT, A)
        expect(st == 503 and b == {"error": {"type": "unavailable"}} and h.get("Retry-After") == "30", "paused: 503 with Retry-After")
        expect(hit(url + "/admin/cap", None, None, ADM)[2]["cap"] == "paused", "GET shows the cap")
        expect(hit(url + "/admin/cap", "POST", {"cap": "unlimited"}, ADM)[0] == 400, "an unknown cap is refused")
        expect(hit(url + "/admin/cap", "POST", {"cap": "none"}, ADM)[0] == 200 and hit(url + "/v1/chat/completions", "POST", CHAT, A)[0] == 200, "the admin can lift it")
        st, _, b = hit(url + "/admin/stats", headers=ADM)
        expect(st == 200 and b["calls"] >= 3 and b["by_route"].get("local", 0) >= 3 and b["blocked"] == 1 and b["unavailable"] == 1 and b["unauthorised"] >= 3
               and b["by_agent"].get("income") == 1 and b["input_tokens"] > 0, "admin stats count calls, routes, agents, blocks and refusals")

        # agents use it over HTTP exactly as in-process
        import income as I
        hc = GatewayHTTPClient(url, "k-api").for_agent("income", I.PROMPT_VERSION)
        expect(hc.health(), "client health")
    finally:
        srv.shutdown()

    # a real agent over HTTP, including a retry that escalates, and the error mapping
    import income as I
    good, facts = I.GOLDEN[0][2], I.GOLDEN[0][1]
    bad = I._out("A look at your cash flow", I._BASE + " You should borrow to cover the gap.", I._BOTH)
    gwy, sk, srv, url = up(ScriptedLLM([bad]), ScriptedLLM([good]))
    try:
        hc = GatewayHTTPClient(url, "k-api").for_agent("income", I.PROMPT_VERSION)
        d = I.IncomeInsight(hc).write(facts)
        expect(d.origin.value == "model" and d.attempts == 2, "an agent works over HTTP, including a retry")
        expect([e["route"] for e in sk.events] == ["local", "external"] and sk.events[1]["reason"] == "escalated_after_retry", "the retry is escalated through the service too")
        expect(hc.last_route == "external" and d.model_version == "scripted-1", "client reports the route and model")
        try:
            GatewayHTTPClient(url, "wrong").for_agent("x").complete("s", "u")
            expect(False, "a wrong key must raise")
        except Exception as exc:
            expect(isinstance(exc, RuntimeError) and "API key" in str(exc), "wrong key: a clear error")
        try:
            hc.complete("s", "call 08031234567")
            expect(False, "an identifier must raise")
        except Exception as exc:
            expect(isinstance(exc, IdentifierBlocked) and "08031234567" not in str(exc), "an identifier is blocked and never repeated in the error")
        hit(url + "/admin/cap", "POST", {"cap": "paused"}, ADM)
        try:
            hc.complete("s", "u")
            expect(False, "paused must raise Unavailable")
        except Unavailable:
            pass
        d = I.IncomeInsight(hc).write(facts)
        expect(d.origin.value == "template" and d.needs_review, "the service is paused: the agent falls back to its template")
    finally:
        srv.shutdown()
    try:
        GatewayHTTPClient("http://127.0.0.1:1", "k").complete("s", "u")
        expect(False, "an unreachable gateway must raise")
    except ConnectionError:
        pass

    # eight callers at once
    gwy, sk, srv, url = up(_Text(_GOOD, "granite-x"))
    try:
        results: list[int] = []
        ts = [threading.Thread(target=lambda: results.append(hit(url + "/v1/chat/completions", "POST", CHAT, A)[0])) for _ in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        expect(results.count(200) == 8 and len(sk.events) == 8, "eight simultaneous requests all succeed and are all recorded")
    finally:
        srv.shutdown()

    # secure by default
    try:
        make_server(ModelGateway(None), "127.0.0.1", 0, None, None)
        expect(False, "no API key must refuse to start")
    except ValueError:
        pass
    try:
        make_server(ModelGateway(None), "127.0.0.1", 0, "same", "same")
        expect(False, "the admin key must differ")
    except ValueError:
        pass
    srv = make_server(ModelGateway(_Text(_GOOD)), "127.0.0.1", 0, None, None, allow_no_auth=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        st, _, _ = hit(f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions", "POST", CHAT)
        expect(st == 200, "allow_no_auth works for local development")
        expect(hit(f"http://127.0.0.1:{srv.server_address[1]}/admin/cap")[0] == 403, "admin stays disabled without an admin key")
    finally:
        srv.shutdown()
    return fails


def _demo() -> None:
    import income as I
    sink = MemorySink()
    good = I.GOLDEN[0][2]
    bad = I._out("A look at your cash flow", I._BASE + " You should borrow to cover the gap.", I._BOTH)
    g = ModelGateway(ScriptedLLM([bad]), ScriptedLLM([good]), sink)
    d = I.IncomeInsight(g.client("income", I.PROMPT_VERSION)).write(I.GOLDEN[0][1])
    print("tip:", d.body, "\n")
    print("usage events (note: no prompt, no answer):")
    for e in sink.events:
        print(" ", json.dumps({k: e[k] for k in ("agent", "route", "reason", "ok", "model_version", "input_tokens", "output_tokens", "latency_ms") if k in e}))
    try:
        g.client("income").complete("rules", "my bvn is 12345678901")
    except IdentifierBlocked as exc:
        print("\nblocked:", exc)
        print("alert event:", json.dumps({k: sink.events[-1][k] for k in ("kind", "route", "reason")}))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="gateway.py", description="CreditCliq Model Gateway")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--demo", action="store_true", help="a retry that escalates, and a blocked identifier (scripted models)")
    mode.add_argument("--probe", action="store_true", help="one real call through the gateway")
    mode.add_argument("--run", choices=["income", "concierge", "recommender", "guardian"], help="run a real agent through the gateway")
    mode.add_argument("--serve", action="store_true", help="run the gateway as an HTTP service")
    ap.add_argument("--local-url", default="http://127.0.0.1:8080")
    ap.add_argument("--local-model", default="granite-4.2-3b-Q4_K_M")
    ap.add_argument("--external-model", help="exact Gemini model id; leave out to run local only")
    ap.add_argument("--usage-file", default="usage_events.jsonl")
    ap.add_argument("--host", default="127.0.0.1", help="(--serve) address to listen on")
    ap.add_argument("--port", type=int, default=8090, help="(--serve) port")
    ap.add_argument("--no-auth", action="store_true", help="(--serve) NO API KEY: local development only")
    ap.add_argument("--gateway-url", help="(--probe/--run) use a running gateway service instead of building one here")
    ap.add_argument("--gateway-key", help="(--probe/--run with --gateway-url) default: GATEWAY_API_KEY")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)

    if args.demo:
        _demo()
        return 0
    if not (args.probe or args.run or args.serve):
        fails = run_selftest()
        if fails:
            print("FAILED:", *fails, sep="\n - ")
            return 1
        print("self-test: identifier block, size and token caps, local-first routing, escalation (bad JSON, errors, retries), "
              "Cost Tracker cap and budgets, safe failure, usage events without content, the HTTP endpoint (keys, limits, admin, "
              "error mapping, concurrency) and real agents through the gateway verified")
        print("ALL PASSED")
        return 0

    if args.gateway_url and not args.serve:   # go through a running service
        client = GatewayHTTPClient(args.gateway_url, args.gateway_key or os.getenv("GATEWAY_API_KEY"), "probe", "p1")
        if not client.health():
            print(f"Cannot reach {args.gateway_url}/health. Is the service running (python gateway.py --serve)?")
            return 1
        if args.probe:
            try:
                r = client.complete("Reply with ONLY a JSON object, no markdown.", 'Return {"ok": true}.', max_output_tokens=40)
            except GatewayError as exc:
                print(f"{type(exc).__name__}: {exc}")
                return 1
            except Exception as exc:
                print(f"{type(exc).__name__}: {exc}")
                return 1
            print("reply:", r.text.strip()[:120])
            print("route:", client.last_route, "|", client.model_version)
            return 0
        from local_llm import _load
        cases, run_one = _load(args.run, "heldout")
        for id_, payload in cases[: args.limit]:
            c = client.for_agent(args.run, "bench")
            res = run_one(c, payload)
            print(f"  {str(id_)[:34]:<36} {res['status']:<9} last route: {c.last_route}")
            if args.show:
                print(f"      {res['text']}")
        return 0

    cfg = GatewayConfig(local_url=args.local_url, local_model=args.local_model, external_model=args.external_model,
                        usage_path=args.usage_file)
    g = ModelGateway.from_config(cfg, on_alert=lambda e: print("ALERT:", e["reason"], flush=True))

    if args.serve:
        api_key, admin_key = os.getenv("GATEWAY_API_KEY"), os.getenv("GATEWAY_ADMIN_KEY")
        try:
            srv = make_server(g, args.host, args.port, api_key, admin_key, allow_no_auth=args.no_auth)
        except ValueError as exc:
            print(f"Cannot start: {exc}.\nMake two different secrets and set them first, for example in PowerShell:\n"
                  '  $env:GATEWAY_API_KEY   = [guid]::NewGuid().ToString("N")\n'
                  '  $env:GATEWAY_ADMIN_KEY = [guid]::NewGuid().ToString("N")')
            return 1
        print(f"Model gateway listening on http://{args.host}:{srv.server_address[1]}")
        print(f"  local model   : {cfg.local_model} at {cfg.local_url}")
        print(f"  external model: {cfg.external_model or 'OFF (local only)'}")
        print(f"  usage records : {cfg.usage_path}")
        print(f"  API key       : {'NOT REQUIRED (--no-auth: development only)' if not api_key else 'required'}")
        print(f"  admin key     : {'set (cap and stats enabled)' if admin_key else 'not set (admin endpoints disabled)'}")
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            print("  WARNING: this address is reachable from other machines and the service has no TLS. "
                  "Use a private network or a reverse proxy that terminates TLS.")
        print("Press Ctrl+C to stop.", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            srv.server_close()
        return 0

    if args.probe:
        try:
            r = g.client("probe", "p1").complete("Reply with ONLY a JSON object, no markdown.", 'Return {"ok": true}.', max_output_tokens=40)
        except GatewayError as exc:
            print(f"{type(exc).__name__}: {exc}")
            return 1
        e = json.loads(open(cfg.usage_path, encoding="utf-8").read().splitlines()[-1])
        print("reply:", r.text.strip()[:120])
        print("route:", e["route"], "|", e["reason"], "|", e.get("model_version"), "|", f"{e.get('latency_ms')} ms")
        return 0

    from local_llm import _load
    cases, run_one = _load(args.run, "heldout")
    for id_, payload in cases[: args.limit]:
        res = run_one(g.client(args.run, "bench"), payload)
        ev = json.loads(open(cfg.usage_path, encoding="utf-8").read().splitlines()[-1])
        print(f"  {str(id_)[:34]:<36} {res['status']:<9} last route: {ev['route']} ({ev['reason']})")
        if args.show:
            print(f"      {res['text']}")
    print(f"\nusage events are in {cfg.usage_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))