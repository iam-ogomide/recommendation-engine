"""CreditCliq Playground: try every AI Engineering agent by hand, the way the app would use them.

Two kinds of agent, so two ways to use them:
  * The Concierge is a CHAT. Type a question like a subscriber.
  * Recommendation, Identity Guardian and Income Insight are EVENT writers. Nobody chats with them: the platform
    notices something (score fell, new loan on the report, income summary ready) and they write ONE notification.
    So you trigger an event and see the notification, then the Policy Reviewer's verdict on it.

Pick the model path (same three options as the other files):
  python test.py --gateway-url http://127.0.0.1:8090     through the running gateway (production shape)
  python test.py --local-url http://127.0.0.1:8080       straight to llama-server
  python test.py --model <exact gemini id>               straight to Gemini
  python test.py --no-model                              no model at all: shows the safe template fallbacks
Add  --kb kb.sqlite  to give the Concierge the knowledge base.   Keys come from the environment or a .env file.

Inside the playground type /help.   Try  /day  first: it runs a whole day for one subscriber.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import textwrap
import time
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import concierge as C
import guardian as G
import income as I
import recommender as R
from recommender import GeminiClient, ScriptedLLM
from reviewer import Reviewer

HELP = """\
Just type a question to talk to the CONCIERGE (a subscriber chat). Quick replies can be picked by number.

Trigger an EVENT (the writer agent makes a notification, then the Policy Reviewer judges it):
  /drop 650 610 [crc|firstcentral|creditregistry]    score fell from 650 to 610           -> Recommendation Agent
  /low 540 [600]                                     score is 540, the line is 600         -> Recommendation Agent
  /late 3 1 [90]                                     3 late/missed, 1 non-performing, 90 days -> Recommendation Agent
  /newitem [LOAN|CREDIT_CARD|OVERDRAFT|MORTGAGE|OTHER|ENQUIRY] [crc|firstcentral|creditregistry] [band]
                                                     a new item on the report              -> Identity Guardian
                                                     bands: UNDER_100K 100K_TO_500K 500K_TO_1M 1M_TO_5M OVER_5M
  /income 450000 32 [180000]                         average monthly income, DBR %, recurring -> Income Insight

Everything together and the safety net:
  /day            one subscriber's day: every event above, then a chat question
  /check <json>   paste a model answer and see the guardrails judge it against the LAST event's facts
  /redteam        feed deliberately bad answers (invented numbers, promises, lender names, phone numbers) to the guardrails

Chat settings:  /plan entry|standard|top   /quota N|none   /scenario fell|rose|flat   /reset
Other:          /show (toggle raw JSON)   /help   /quit
"""

BUREAUS = {"crc": "CRC", "firstcentral": "FIRST_CENTRAL", "fc": "FIRST_CENTRAL", "creditregistry": "CREDIT_REGISTRY", "cr": "CREDIT_REGISTRY"}


# =============================================================================
# model path
# =============================================================================

def _load_dotenv() -> None:
    p = os.path.join(HERE, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_client_factory(args):
    """Returns (factory(agent, prompt_version, reviewer=False) -> client, label)."""
    if args.no_model:
        return (lambda agent, pv, reviewer=False: ScriptedLLM([])), "no model (template fallbacks only)"
    if args.gateway_url:
        from gateway import GatewayHTTPClient
        gw = GatewayHTTPClient(args.gateway_url, os.getenv("GATEWAY_API_KEY"))
        if not gw.health():
            raise SystemExit(f"Cannot reach {args.gateway_url}/health. Start it: python gateway.py --serve --external-model <id>")
        return (lambda agent, pv, reviewer=False: gw.for_agent("reviewer" if reviewer else agent, "review" if reviewer else pv)), "the gateway"
    if args.local_url:
        from local_llm import LocalLLMClient
        return (lambda agent, pv, reviewer=False: LocalLLMClient(args.local_url, model=args.local_model,
                                                                 api_key=os.getenv("LLAMA_API_KEY"))), "the local model"
    if args.model:
        return (lambda agent, pv, reviewer=False: GeminiClient(args.model, temperature=0.0) if reviewer else GeminiClient(args.model)), "Gemini"
    raise SystemExit("Choose a model path: --gateway-url URL, --local-url URL, --model <gemini id>, or --no-model")


# =============================================================================
# showing a notification the way the phone would
# =============================================================================

def _box(title: str, lines: list[str], footer: str) -> None:
    width = 62
    print("  ┌" + "─" * width)
    print(f"  │ {title}")
    for ln in lines:
        for w in textwrap.wrap(ln, width - 2) or [""]:
            print(f"  │ {w}")
    print("  ├" + "─" * width)
    for w in textwrap.wrap(footer, width - 2):
        print(f"  │ {w}")
    print("  └" + "─" * width)


def show_draft(name: str, draft, review, secs: float, route: str | None, raw: bool) -> None:
    acts = "   ".join(f"[{a.label}]" for a in draft.actions)
    verdict = getattr(review.verdict, "value", review.verdict)
    held = draft.needs_review or verdict in ("reject", "unavailable")
    foot = (f"{name} | wrote it: {draft.origin.value}, {draft.attempts} attempt(s) | reviewer: {verdict} | "
            f"{'WOULD BE HELD FOR A HUMAN' if held else 'would publish on its own'} | {secs:.1f}s"
            + (f" | route: {route}" if route else ""))
    _box(f"PUSH  {draft.title}", [draft.body, "", acts], foot)
    codes = getattr(draft, "failed_attempt_codes", None)
    if codes:
        print(f"    guardrails rejected the model's earlier tries: {codes}")
    for v in getattr(draft, "review_reasons", None) or []:
        print(f"    last violation: {getattr(v, 'code', v)} {getattr(v, 'detail', '') or ''}")
    if verdict in ("reject", "unavailable"):
        for r in review.reasons:
            print(f"    reviewer says {r.code}: {r.why}  (quoted: {r.quote!r}, found in draft: {r.quote_verified})")
    if raw:
        print("    facts:", json.dumps(draft.trigger_facts, sort_keys=True))


# =============================================================================
# the playground
# =============================================================================

class Playground:
    def __init__(self, factory, kb_args, raw: bool = False):
        self.f, self.raw = factory, raw
        handlers = None
        if kb_args.kb:
            from KB import KnowledgeBase, KnowledgeSearchTool, make_embedder
            handlers = {"knowledge.search": KnowledgeSearchTool(KnowledgeBase(kb_args.kb, make_embedder(kb_args.embedder, kb_args.model_dir)))}
        self.fx = dict(C.SCENARIOS["fell"])
        cw = factory("concierge", C.PROMPT_VERSION)
        self.cw = cw
        self.chat = C.ChatSession(C.Concierge(cw, fixtures=self.fx, handlers=handlers), C.PlanTier.STANDARD, None, self.fx)
        self.chat_reviewer = C.ConciergeReviewer(factory("concierge", "", reviewer=True))
        self.last: tuple[str, dict] | None = None   # (agent, facts) of the last event, for /check

    # ---- events -----------------------------------------------------------
    def _run(self, name, facts, writer_cls, writer_pv, reviewer_cls, **kw) -> None:
        w = self.f(name, writer_pv)
        t0 = time.time()
        try:
            draft = writer_cls(w, **kw).write(facts)
        except Exception as exc:   # invalid FACTS: the agent refuses them before any model is called
            errs = getattr(exc, "errors", None)
            msg = "; ".join(str(e.get("msg", "")).replace("Value error, ", "") for e in errs()) if callable(errs) else str(exc)
            print(f"  The facts were refused before any model was called: {msg[:200]}")
            return
        review = reviewer_cls(self.f(name, "", reviewer=True)).review(draft)
        show_draft(name, draft, review, time.time() - t0, getattr(w, "last_route", None), self.raw)
        self.last = (name, facts)

    def recommender(self, facts):
        self._run("recommender", facts, R.Recommender, R.PROMPT_VERSION, Reviewer)

    def guardian(self, facts):
        self._run("guardian", facts, G.Guardian, G.GUARDIAN_PROMPT_VERSION, G.GuardianReviewer, always_review=False)

    def income(self, facts):
        self._run("income", facts, I.IncomeInsight, I.PROMPT_VERSION, I.IncomeReviewer)

    # ---- facts from what you type ----------------------------------------
    @staticmethod
    def _bureau(tok: list[str], default="FIRST_CENTRAL") -> str:
        for t in tok:
            if t.lower() in BUREAUS:
                return BUREAUS[t.lower()]
        return default

    def cmd_drop(self, a):
        prev, cur = int(a[0]), int(a[1])
        today = date.today()
        self.recommender(R._sd(prev, cur, (today - timedelta(days=30)).isoformat(), today.isoformat(), self._bureau(a[2:])))

    def cmd_low(self, a):
        self.recommender(R._ls(int(a[0]), int(a[1]) if len(a) > 1 else 600))

    def cmd_late(self, a):
        self.recommender(R._pr(int(a[0]), int(a[1]), int(a[2]) if len(a) > 2 else 90))

    def cmd_newitem(self, a):
        item = next((t.upper() for t in a if t.upper() in ("LOAN", "CREDIT_CARD", "OVERDRAFT", "MORTGAGE", "OTHER", "ENQUIRY")), "LOAN")
        band = next((t.upper() for t in a if t.upper() in ("UNDER_100K", "100K_TO_500K", "500K_TO_1M", "1M_TO_5M", "OVER_5M")), "UNKNOWN")
        today = date.today()
        self.guardian(G._f(self._bureau(a, "CRC"), item, (today - timedelta(days=2)).isoformat(), today.isoformat(), band))

    def cmd_income(self, a):
        avg, dbr = int(a[0]), int(a[1])
        rec = int(a[2]) if len(a) > 2 else int(avg * 0.4)
        end = date.today().replace(day=1) - timedelta(days=1)
        start = (end.replace(day=1) - timedelta(days=62)).replace(day=1)
        self.income(I._inc(avg=avg, tin=avg * 3, tout=int(avg * 3 * 0.9), recurring=rec, dbr=dbr, start=start.isoformat(), end=end.isoformat()))

    # ---- chat -------------------------------------------------------------
    def ask(self, line: str) -> None:
        s = self.chat
        if s.pending_refresh and line.lower() in ("yes", "y", "no", "n"):
            if line.lower() in ("yes", "y"):
                s.confirm_refresh()
                print("  assistant> Confirmed. In the app this starts the paid refresh. Nothing was run here.")
            else:
                s.pending_refresh = False
                print("  assistant> Okay, I have not refreshed anything.")
            return
        if line.isdigit() and s.last and 1 <= int(line) <= len(s.last.suggestions):
            line = s.last.suggestions[int(line) - 1]
            print(f"  you> {line}")
        t0 = time.time()
        reply = s.ask(line)
        secs = time.time() - t0
        for w in textwrap.wrap(reply.answer, 76):
            print(f"  assistant> {w}" if w is reply.answer else f"             {w}")
        if reply.citations:
            print("     source: " + "; ".join(C._fmt_cite(c) for c in reply.citations))
        if reply.guidance:
            print("     guidance used: " + ", ".join(reply.guidance))
        if reply.input_required:
            print("     Confirm the paid refresh? (yes / no)")
        if reply.handoff:
            print("     (this needs a human support agent)")
        if reply.suggestions:
            print("     quick replies: " + "   ".join(f"[{i}] {sg}" for i, sg in enumerate(reply.suggestions, 1)))
        rv = self.chat_reviewer.review(C.review_view(reply))
        verdict = getattr(rv.verdict, "value", rv.verdict)
        print(f"     [{reply.origin}, intent={reply.intent}, tools={reply.tools_called or 'none'}, attempts={reply.attempts}, "
              f"reviewer={verdict}, {secs:.1f}s" + (f", route={self.cw.last_route}" if getattr(self.cw, 'last_route', None) else "") + "]")
        if reply.failed_attempt_codes:
            print(f"     guardrails rejected the model's earlier tries: {reply.failed_attempt_codes}")
        for r in (rv.reasons if verdict in ("reject", "unavailable") else []):
            print(f"     reviewer says {r.code}: {r.why}")

    # ---- safety net -------------------------------------------------------
    def _checker(self, agent: str, facts: dict):
        if agent == "recommender":
            return lambda raw: R.check_output(raw, R.parse_facts(facts))
        if agent == "guardian":
            return lambda raw: G.check_alert(raw, G.parse_facts(facts))
        return lambda raw: I.check_output(raw, I.parse_facts(facts))

    def cmd_check(self, text: str) -> None:
        if not self.last:
            print("  Run an event first (for example /drop 650 610): the check needs that event's facts.")
            return
        agent, facts = self.last
        res = self._checker(agent, facts)(text)
        if res.ok:
            print(f"  ACCEPTED by the {agent} guardrails: this would be shown to the subscriber.")
        else:
            print(f"  REJECTED by the {agent} guardrails:")
            for v in res.violations:
                print(f"    {v.code}: {v.detail or ''}")

    def cmd_redteam(self) -> None:
        facts = R._sd(650, 610, "2026-10-01", "2026-10-31")
        rpt = [("View my report", "credit_report")]
        cases = [
            ("a correct message", R._out("Your credit score dropped", "Your FirstCentral score fell from 650 to 610, a drop of 40 points, since 1 Oct.", rpt)),
            ("an invented number", R._out("Your credit score dropped", "Your FirstCentral score fell from 650 to 610, a drop of 99 points.", rpt)),
            ("a promise", R._out("Your credit score dropped", "Your score fell from 650 to 610. Fix it and you will be approved for a loan.", rpt)),
            ("a lender name", R._out("Your credit score dropped", "Your score fell from 650 to 610. Carbon can offer you a loan today.", rpt)),
            ("a phone number", R._out("Your credit score dropped", "Your score fell from 650 to 610. Call 08012345678 now.", rpt)),
            ("a screen that does not exist", R._out("Your credit score dropped", "Your score fell from 650 to 610.", [("Get a loan", "loan_marketplace")])),
            ("not JSON at all", "Sure! Your score went down, here is some advice."),
        ]
        print("  Each answer below is what a misbehaving model might write for: score fell 650 -> 610 (FirstCentral).")
        for label, raw in cases:
            res = R.check_output(raw, R.parse_facts(facts))
            codes = "" if res.ok else "  " + ", ".join(dict.fromkeys(v.code for v in res.violations))
            print(f"  {'ACCEPTED' if res.ok else 'REJECTED':<9} {label}{codes}")

    def cmd_day(self) -> None:
        print("  One subscriber's day (synthetic data). The daily job finds these events; each writer drafts a notification.\n")
        for title, fn, a in [("08:00  score fell", self.cmd_drop, ["650", "610"]), ("08:01  repayments slipped", self.cmd_late, ["3", "1"]),
                             ("08:02  a new loan appeared on the report", self.cmd_newitem, ["LOAN", "crc", "100K_TO_500K"]),
                             ("08:03  income summary ready", self.cmd_income, ["450000", "32"])]:
            print(f"  {title}")
            fn(a)
            print()
        print("  19:30  the subscriber opens the app and asks the Concierge:")
        print("  you> Why did my score go down?")
        self.ask("Why did my score go down?")

    # ---- loop -------------------------------------------------------------
    def handle(self, line: str) -> bool:
        if not line.startswith("/"):
            self.ask(line)
            return True
        cmd, _, rest = line.partition(" ")
        try:
            a = shlex.split(rest)
            if cmd in ("/quit", "/exit"):
                return False
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/show":
                self.raw = not self.raw
                print(f"  raw facts {'on' if self.raw else 'off'}")
            elif cmd == "/day":
                self.cmd_day()
            elif cmd == "/redteam":
                self.cmd_redteam()
            elif cmd == "/check":
                self.cmd_check(rest.strip())
            elif cmd in ("/drop", "/low", "/late", "/newitem", "/income"):
                getattr(self, "cmd_" + cmd[1:])(a)
            elif cmd == "/plan" and a and a[0] in C.PLAN_LABEL:
                self.chat.plan = C.PlanTier(a[0]); print(f"  plan is now {C.PLAN_LABEL[a[0]]}")
            elif cmd == "/quota":
                self.chat.quota = None if (a and a[0] == "none") else int(a[0]); print(f"  refreshes left: {self.chat.quota}")
            elif cmd == "/scenario" and a and a[0] in C.SCENARIOS:
                self.chat.set_scenario(a[0]); print(f"  demo data: {a[0]} (conversation cleared)")
            elif cmd == "/reset":
                self.chat.reset(); print("  conversation cleared")
            else:
                print("  Unknown or incomplete command. Type /help.")
        except (IndexError, ValueError):
            print("  That command needs numbers, for example:  /drop 650 610   /low 540   /late 3 1   /income 450000 32")
        return True


def main(argv: list[str]) -> int:
    for stream in (sys.stdout, sys.stderr):   # Windows consoles default to a code page that cannot print the box or the naira sign
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    _load_dotenv()
    ap = argparse.ArgumentParser(description="CreditCliq agent playground")
    ap.add_argument("--gateway-url")
    ap.add_argument("--local-url")
    ap.add_argument("--local-model", default="granite-4.2-3b-Q4_K_M")
    ap.add_argument("--model", help="exact Gemini model id")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--kb")
    ap.add_argument("--embedder", choices=["hash", "e5"], default="hash")
    ap.add_argument("--model-dir")
    args = ap.parse_args(argv)
    factory, via = make_client_factory(args)
    pg = Playground(factory, args)
    print(f"CreditCliq playground. Answers come from {via}. Data is synthetic. Type /day for a full demo, /help for commands.")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if line and not pg.handle(line):
            return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))