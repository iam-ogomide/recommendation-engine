"""CreditCliq local inference: the client AND the benchmark, in one file.  Talks to a llama.cpp `llama-server`.

PART 1  LocalLLMClient   implements the SAME interface as GeminiClient and ScriptedLLM (recommender.LLMClient), so
                         every agent (recommender, guardian, income, concierge, reviewer) can use a local model with
                         no other change:

    from local_llm import LocalLLMClient
    llm = LocalLLMClient("http://127.0.0.1:8080", model="granite-4.2-3b-Q4_K_M", api_key="...")
    Concierge(llm).answer({...})

PART 2  the benchmark    runs each agent's held-out cases through the REAL agent (real prompts, real checks, real
                         retry) with the local model as the writer, then reports first-try / after-retry /
                         template-fallback counts, latency, speed, and which rules the model trips on. Mirrors the
                         local-inference SKILL.md: same prompts as the golden set, pass the number-match and format
                         checks, compare speed and fallback rate.

Run:
  python local_llm.py --probe                          one tiny call: is the server, JSON mode and speed sane?
  python local_llm.py --agent income --limit 3 --show  one agent, a few cases, print the outputs
  python local_llm.py --agent all --runs 3 --out granite42_q4.json
  python local_llm.py --selftest                       tests this file against a fake server (no model needed)

API key: the llama-server key is a secret YOU choose when you start the server (--api-key). Give this file the same
value with --api-key or the LLAMA_API_KEY variable (a .env file works too). In production this client is replaced by
the Model gateway client; the gateway owns routing, redaction and caching.

The client: standard library only (no new dependency). JSON mode (grammar-constrained) by default, prompt caching,
per-call timings kept in `.stats`, and errors that name only the HTTP status or error type: the prompt and the
server's reply body never appear in an exception message, so nothing sensitive leaks into a log through one.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable

from recommender import LLMResponse, TEMPERATURE


# =============================================================================
# PART 1: THE CLIENT
# =============================================================================

class LocalLLMClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", model: str = "local", api_key: str | None = None,
                 temperature: float = TEMPERATURE, timeout: float = 300.0, seed: int | None = None,
                 json_mode: str = "object", no_thinking: bool = True):
        if json_mode not in ("object", "schema", "off"):
            raise ValueError("json_mode must be 'object', 'schema' or 'off'")
        if not model or any(ch.isspace() for ch in model):
            raise ValueError(f"Invalid model id {model!r}: use the exact GGUF/model name, no spaces.")
        self._base = base_url.rstrip("/")
        self.model_version = model          # pin an EXACT id, e.g. the GGUF file name
        self._key = api_key
        self._temperature = temperature
        self._timeout = timeout
        self._seed = seed
        self._json_mode = json_mode
        self._no_thinking = no_thinking
        self.stats: list[dict[str, Any]] = []

    # ---- plumbing ---------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._key:
            h["Authorization"] = f"Bearer {self._key}"
        return h

    def _request(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self._base + path, data=data, headers=self._headers(),
                                     method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:   # status only: the body can echo the prompt
            raise RuntimeError(f"local server returned HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            raise ConnectionError(f"cannot reach local server ({type(exc.reason).__name__})") from None
        except json.JSONDecodeError:
            raise RuntimeError("local server returned a non-JSON reply") from None

    def health(self) -> bool:
        try:
            return self._request("/health", timeout=5.0).get("status") == "ok"
        except Exception:
            return False

    def wait_ready(self, seconds: float = 120.0) -> bool:
        end = time.time() + seconds
        while time.time() < end:
            if self.health():
                return True
            time.sleep(1.0)
        return False

    # ---- the agent interface (same as GeminiClient / ScriptedLLM) -----------------
    def complete(self, system: str, user: str, *, max_output_tokens: int | None = None,
                 json_schema: dict | None = None) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model_version,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self._temperature,
            "stream": False,
            "cache_prompt": True,
        }
        if max_output_tokens is not None:
            body["max_tokens"] = max_output_tokens
        if self._seed is not None:
            body["seed"] = self._seed
        if self._json_mode == "schema" and json_schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "reply", "strict": True, "schema": json_schema}}
        elif self._json_mode in ("object", "schema"):
            body["response_format"] = {"type": "json_object"}
        if self._no_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}   # ignored by templates without a thinking mode

        t0 = time.perf_counter()
        reply = self._request("/v1/chat/completions", body)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        choices = reply.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""   # "" fails the format check -> normal retry
        usage = reply.get("usage") or {}
        timings = reply.get("timings") or {}
        in_tok = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
        out_tok = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
        self.stats.append({
            "wall_ms": wall_ms,
            "prompt_n": in_tok, "predicted_n": out_tok,
            "prompt_ms": float(timings.get("prompt_ms") or 0.0),
            "predicted_ms": float(timings.get("predicted_ms") or 0.0),
            "predicted_per_second": float(timings.get("predicted_per_second") or 0.0),
            "prompt_per_second": float(timings.get("prompt_per_second") or 0.0),
            "cache_n": int(timings.get("cache_n") or 0),
            "finish_reason": choices[0].get("finish_reason"),
        })
        return LLMResponse(text=text, input_tokens=in_tok, output_tokens=out_tok)

    # ---- benchmark helpers ---------------------------------------------------------
    def summary(self, since: int = 0) -> dict[str, Any]:
        rows = self.stats[since:]
        if not rows:
            return {"calls": 0}
        gen = [r["predicted_per_second"] for r in rows if r["predicted_per_second"] > 0]
        pre = [r["prompt_per_second"] for r in rows if r["prompt_per_second"] > 0]
        walls = sorted(r["wall_ms"] for r in rows)
        return {
            "calls": len(rows),
            "prompt_tokens": sum(r["prompt_n"] for r in rows),
            "output_tokens": sum(r["predicted_n"] for r in rows),
            "cached_prompt_tokens": sum(r["cache_n"] for r in rows),
            "gen_tok_per_s_median": round(statistics.median(gen), 1) if gen else None,
            "prompt_tok_per_s_median": round(statistics.median(pre), 1) if pre else None,
            "wall_ms_p50": round(walls[len(walls) // 2]),
            "wall_ms_p95": round(walls[min(len(walls) - 1, int(len(walls) * 0.95))]),
            "truncated": sum(1 for r in rows if r["finish_reason"] == "length"),
        }


# =============================================================================
# PART 2: THE BENCHMARK
# =============================================================================

FALLBACK_BAR = 0.15   # same bar the agents' --live modes use
AGENTS = ("recommender", "guardian", "income", "concierge")


# =============================================================================
# One adapter per agent: cases, how to run one case, how to read the result
# =============================================================================

def _val(x: Any) -> str:
    return str(getattr(x, "value", x))


def _load(agent: str, which: str) -> tuple[list[tuple[str, Any]], Callable[[Any, Any], dict]]:
    """Returns (cases, run_one). run_one(llm, case_payload) -> {status, attempts, codes, text}."""
    if agent == "recommender":
        import recommender as R
        cases = list(R.HELD_OUT) + ([(i, f) for i, f, _ in R.GOLDEN] if which == "all" else [])

        def run(llm, facts):
            d = R.Recommender(llm).write(facts)
            return _draft_result(d, f"{d.title} | {d.body}")
        return cases, run

    if agent == "guardian":
        import guardian as G
        cases = list(G.HELD_OUT) + ([(i, f) for i, f, _ in G.GOLDEN] if which == "all" else [])

        def run(llm, facts):
            d = G.Guardian(llm).write(facts)
            return _draft_result(d, f"{d.title} | {d.body}")
        return cases, run

    if agent == "income":
        import income as I
        cases = list(I.HELD_OUT) + ([(i, f) for i, f, _ in I.GOLDEN] if which == "all" else [])

        def run(llm, facts):
            d = I.IncomeInsight(llm).write(facts)
            return _draft_result(d, f"{d.title} | {d.body}")
        return cases, run

    if agent == "concierge":
        import concierge as C
        cases = [(q[:40], (q, plan, fx)) for q, plan, fx in C.HELD_OUT]
        if which == "all":
            cases += [(f"golden-{i}", (q, plan, C.DEFAULT_FIXTURES)) for i, q, plan, _ in C.GOLDEN]

        def run(llm, payload):
            q, plan, fx = payload
            r = C.Concierge(llm, fixtures=fx).answer({"question": q, "plan": plan})
            if r.origin == "refusal":
                return {"status": "refused", "attempts": 0, "codes": [], "text": r.answer}
            status = "template" if r.origin == "template" else ("first" if r.attempts == 1 else "retry")
            return {"status": status, "attempts": r.attempts, "codes": [c for a in r.failed_attempt_codes for c in a],
                    "text": r.answer}
        return cases, run

    raise ValueError(f"unknown agent {agent!r}")


def _draft_result(d: Any, text: str) -> dict:
    if _val(d.origin) == "template":
        status = "template"
    else:
        status = "first" if d.attempts == 1 else "retry"
    codes = [c for attempt in getattr(d, "failed_attempt_codes", []) for c in attempt]
    return {"status": status, "attempts": d.attempts, "codes": codes, "text": text}


# =============================================================================
# The benchmark
# =============================================================================

def run_agent(agent: str, llm: LocalLLMClient, which: str, runs: int, limit: int | None, show: bool) -> dict:
    cases, run_one = _load(agent, which)
    if limit:
        cases = cases[:limit]
    rows: list[dict] = []
    print(f"\n== {agent}: {len(cases)} cases x {runs} run(s) ==")
    for r in range(runs):
        for id_, payload in cases:
            mark = len(llm.stats)
            t0 = time.perf_counter()
            res = run_one(llm, payload)
            wall = (time.perf_counter() - t0) * 1000
            s = llm.summary(mark)
            res.update({"id": id_, "run": r + 1, "wall_ms": round(wall), "calls": s.get("calls", 0),
                        "out_tokens": s.get("output_tokens", 0), "gen_tps": s.get("gen_tok_per_s_median")})
            rows.append(res)
            tps = f"{res['gen_tps']:>5} tok/s" if res["gen_tps"] else "      - tok/s"
            extra = f"  rejected for {sorted(set(res['codes']))}" if res["codes"] else ""
            print(f"  {id_[:34]:<36} {res['status']:<9} {res['wall_ms']:>6} ms  {res['out_tokens']:>4} tok  {tps}{extra}")
            if show:
                print(f"      {res['text']}")
    model_rows = [x for x in rows if x["status"] != "refused"]
    n = len(model_rows)
    counts = Counter(x["status"] for x in rows)
    codes = Counter(c for x in rows for c in x["codes"])
    out = {
        "agent": agent, "cases": len(cases), "runs": runs, "model_turns": n,
        "first": counts["first"], "retry": counts["retry"], "template": counts["template"], "refused": counts["refused"],
        "fallback_rate": round(counts["template"] / n, 3) if n else 0.0,
        "first_try_rate": round(counts["first"] / n, 3) if n else 0.0,
        "top_rejections": codes.most_common(6),
    }
    print(f"  -> first try {out['first']}, after retry {out['retry']}, template fallback {out['template']}"
          + (f", refused by code {out['refused']}" if out["refused"] else "")
          + f"  (fallback rate {out['fallback_rate']:.0%})")
    if codes:
        print("  -> rules the model tripped: " + ", ".join(f"{c} x{k}" for c, k in codes.most_common(6)))
    return out


def probe(llm: LocalLLMClient) -> int:
    """One tiny call so you can see whether the server, JSON mode and speed are sane before benchmarking."""
    print(f"server {llm._base}  model id {llm.model_version}")
    if not llm.wait_ready(5):
        print("FAIL: /health is not ok. Is llama-server running, and is the address right?")
        return 1
    try:
        r = llm.complete('Reply with ONLY a JSON object, no markdown.',
                         'Return {"ok": true, "answer": 2+2 written as a digit}.', max_output_tokens=60)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    s = llm.stats[-1]
    print(f"reply            : {r.text.strip()[:160]!r}")
    try:
        json.loads(r.text)
        print("valid JSON       : yes")
    except json.JSONDecodeError:
        print("valid JSON       : NO  (JSON mode may not be applied by this build: try --json-mode schema or off and compare)")
    print(f"thinking text    : {'YES, found <think> in the reply: turn thinking off in the server or template' if '<think' in r.text.lower() else 'none seen'}")
    print(f"prompt           : {s['prompt_n']} tokens in {s['prompt_ms']:.0f} ms ({s['prompt_per_second']:.1f} tok/s)")
    print(f"generation       : {s['predicted_n']} tokens at {s['predicted_per_second']:.1f} tok/s")
    print(f"round trip       : {s['wall_ms']:.0f} ms   finish_reason={s['finish_reason']}")
    if not s["prompt_ms"] and not s["predicted_per_second"]:
        print("note             : the server sent no 'timings'; speed numbers will be missing from the benchmark.")
    return 0


def prompt_prefix_share(agent: str) -> float | None:
    """How much of the prompt two DIFFERENT calls of the same agent have in common, counted from the start.
    A llama-server can only reuse the cached START of a prompt, so this share is the part it can skip re-reading.
    Returns a number between 0 and 1, or None when the agent has no two comparable cases."""
    from recommender import ScriptedLLM

    def capture(run) -> str:
        llm = ScriptedLLM(["x"])
        run(llm)
        c = llm.calls[0]
        return c["system"] + "\n" + c["user"]

    cases, _ = _load(agent, "heldout")
    runs: list[Callable[[Any], Any]] = []
    if agent == "concierge":
        import concierge as C
        for q, plan, fx in C.HELD_OUT[:2]:
            runs.append(lambda llm, q=q, plan=plan, fx=fx: C.Concierge(llm, fixtures=fx).answer({"question": q, "plan": plan}))
    else:
        by_trigger: dict[str, list[Any]] = {}
        for _, facts in cases:
            by_trigger.setdefault(str(facts.get("trigger_code")), []).append(facts)
        pair = next((v[:2] for v in by_trigger.values() if len(v) >= 2), None)
        if pair is None:
            return None
        mod = {"recommender": "recommender", "guardian": "guardian", "income": "income"}[agent]
        m = __import__(mod)
        cls = {"recommender": "Recommender", "guardian": "Guardian", "income": "IncomeInsight"}[agent]
        for facts in pair:
            runs.append(lambda llm, facts=facts: getattr(m, cls)(llm).write(facts))
    a, b = capture(runs[0]), capture(runs[1])
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n / max(len(a), len(b))


# =============================================================================
# Self-test: a fake llama-server, so the client and the benchmark are tested without a model
# =============================================================================

class _Fake(BaseHTTPRequestHandler):
    mode = "ok"
    content = '{"title": "t", "body": "b", "actions": []}'
    seen: list[dict] = []

    def log_message(self, *a):
        pass

    def _send(self, code, obj=None, raw=None):
        data = raw if raw is not None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.seen.append({"body": body, "auth": self.headers.get("Authorization"), "path": self.path})
        if _Fake.mode == "http500":
            return self._send(500, {"error": "boom: " + json.dumps(body)[:200]})
        if _Fake.mode == "notjson":
            return self._send(200, raw=b"<html>nope</html>")
        content = None if _Fake.mode == "empty" else _Fake.content
        self._send(200, {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
            "timings": {"prompt_n": 120, "prompt_ms": 60.0, "prompt_per_second": 2000.0, "predicted_n": 30,
                        "predicted_ms": 1500.0, "predicted_per_second": 20.0, "cache_n": 100},
        })


def run_selftest() -> list[str]:
    fails: list[str] = []

    def expect(c: bool, what: str):
        if not c:
            fails.append(what)

    # prompts are cache-friendly: the fixed text comes first, so most of a prompt is shared between calls
    for agent, bar in (("income", 0.70), ("recommender", 0.55), ("guardian", 0.55), ("concierge", 0.50)):
        share = prompt_prefix_share(agent)
        expect(share is not None and share >= bar,
               f"{agent}: only {0 if share is None else share:.0%} of the prompt is shared between calls (want >= {bar:.0%}): "
               "put fixed text before the facts")

    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        c = LocalLLMClient(base, model="granite-test", api_key="k123", seed=7)
        expect(c.health(), "health")
        _Fake.seen.clear()
        r = c.complete("SYS", "USER", max_output_tokens=50)
        b, auth = _Fake.seen[0]["body"], _Fake.seen[0]["auth"]
        expect(_Fake.seen[0]["path"] == "/v1/chat/completions", "uses the OpenAI-compatible path")
        expect(b["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}], "system then user message")
        expect(b["cache_prompt"] is True and b["stream"] is False and b["max_tokens"] == 50 and b["seed"] == 7, "cache, no stream, token cap, seed")
        expect(b["response_format"] == {"type": "json_object"}, "JSON mode by default")
        expect(b["chat_template_kwargs"] == {"enable_thinking": False}, "thinking switched off by default")
        expect(auth == "Bearer k123", "api key sent as a bearer token")
        expect(r.text.startswith("{") and r.input_tokens == 120 and r.output_tokens == 30, "reply and token counts parsed")
        expect(c.model_version == "granite-test", "exact model id kept")
        sm = c.summary()
        expect(sm["calls"] == 1 and sm["gen_tok_per_s_median"] == 20.0 and sm["cached_prompt_tokens"] == 100, "timings kept for the benchmark")

        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        c2 = LocalLLMClient(base, model="m", json_mode="schema")
        c2.complete("S", "U", json_schema=schema)
        rf = _Fake.seen[-1]["body"]["response_format"]
        expect(rf["type"] == "json_schema" and rf["json_schema"]["schema"] == schema, "schema mode sends the agent's schema")
        c2.complete("S", "U")
        expect(_Fake.seen[-1]["body"]["response_format"] == {"type": "json_object"}, "schema mode falls back to JSON mode with no schema")
        c3 = LocalLLMClient(base, model="m", json_mode="object")
        c3.complete("S", "U", json_schema=schema)
        expect(_Fake.seen[-1]["body"]["response_format"] == {"type": "json_object"}, "object mode ignores the schema")
        c4 = LocalLLMClient(base, model="m", json_mode="off", no_thinking=False)
        c4.complete("S", "U")
        expect("response_format" not in _Fake.seen[-1]["body"] and "chat_template_kwargs" not in _Fake.seen[-1]["body"], "off means off")
        expect("Authorization" not in c4._headers(), "no key, no auth header")

        _Fake.mode = "empty"
        expect(c.complete("S", "U").text == "", "empty content becomes '' (a normal retry), not a crash")
        _Fake.mode = "http500"
        try:
            c.complete("SECRET-PROMPT-TEXT", "BVN 12345678901")
            expect(False, "HTTP 500 must raise")
        except RuntimeError as exc:
            expect("500" in str(exc) and "SECRET" not in str(exc) and "12345678901" not in str(exc) and "boom" not in str(exc),
                   "an HTTP error names the status only (no prompt or server body in the message)")
        _Fake.mode = "notjson"
        try:
            c.complete("S", "U")
            expect(False, "non-JSON must raise")
        except RuntimeError as exc:
            expect("non-JSON" in str(exc), "non-JSON reply")
        _Fake.mode = "ok"
        try:
            LocalLLMClient("http://127.0.0.1:1", model="m", timeout=2).complete("S", "U")
            expect(False, "unreachable server must raise")
        except ConnectionError:
            pass
        expect(not LocalLLMClient("http://127.0.0.1:1", model="m").health(), "health is False when unreachable")
        for bad in ("", "two words"):
            try:
                LocalLLMClient(base, model=bad)
                expect(False, f"model id {bad!r} must be rejected")
            except ValueError:
                pass

        # benchmark plumbing: a "model" that returns junk must show up as 100% template fallback, per agent
        _Fake.content = "this is not json"
        for agent in AGENTS:
            llm = LocalLLMClient(base, model="junk")
            res = run_agent(agent, llm, "heldout", 1, 2, False)
            n = res["model_turns"]
            expect(res["template"] == n and res["fallback_rate"] == 1.0 and res["first"] == 0, f"{agent}: junk model -> all template (got {res})")
            expect(any(c == "FORMAT_NOT_JSON" for c, _ in res["top_rejections"]) or n == 0, f"{agent}: rejection codes are reported")
        _Fake.content = '{"title": "t", "body": "b", "actions": []}'
    finally:
        srv.shutdown()
    return fails


# =============================================================================
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="local_llm.py", description="Benchmark a local llama-server with the CreditCliq agents")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--probe", action="store_true", help="one tiny call: server, JSON mode and speed")
    ap.add_argument("--agent", choices=[*AGENTS, "all"], default="all")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="granite-4.2-3b-Q4_K_M", help="exact model id, for the record and for comparisons")
    ap.add_argument("--api-key", default=os.getenv("LLAMA_API_KEY"))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--set", dest="which", choices=["heldout", "all"], default="heldout")
    ap.add_argument("--limit", type=int, help="cases per agent (quick check)")
    ap.add_argument("--json-mode", choices=["object", "schema", "off"], default="object")
    ap.add_argument("--no-thinking-flag", action="store_true", help="do not send chat_template_kwargs enable_thinking=false")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--show", action="store_true", help="print every output")
    ap.add_argument("--out", help="write the results to this JSON file")
    args = ap.parse_args(argv)

    if args.selftest:
        fails = run_selftest()
        if fails:
            print("FAILED:", *fails, sep="\n - ")
            return 1
        print("self-test: prompts are cache-friendly, client request shape, JSON/schema/off modes, timings, error handling, and benchmark plumbing verified\nALL PASSED")
        return 0

    llm = LocalLLMClient(args.base_url, model=args.model, api_key=args.api_key, timeout=args.timeout, seed=args.seed,
                         json_mode=args.json_mode, no_thinking=not args.no_thinking_flag)
    if args.probe:
        return probe(llm)
    if not llm.wait_ready(10):
        print(f"Cannot reach {args.base_url}/health. Start llama-server first (see --probe).")
        return 1

    agents = AGENTS if args.agent == "all" else (args.agent,)
    results = [run_agent(a, llm, args.which, args.runs, args.limit, args.show) for a in agents]
    speed = llm.summary()
    print("\n== speed (all calls) ==")
    print(json.dumps(speed, indent=2))
    print("\n== summary ==")
    worst = 0.0
    for r in results:
        worst = max(worst, r["fallback_rate"])
        print(f"  {r['agent']:<12} first-try {r['first_try_rate']:.0%}   fallback {r['fallback_rate']:.0%}   ({r['model_turns']} model turns)")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "base_url": args.base_url, "json_mode": args.json_mode, "set": args.which,
                       "runs": args.runs, "speed": speed, "agents": results}, fh, indent=2)
        print(f"\nsaved {args.out}")
    return 1 if worst > FALLBACK_BAR else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))