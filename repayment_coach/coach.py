"""
coach.py  -  THE REPAYMENT COACH (everything in one file)

What the Repayment Coach does, in simple words:
  It helps a subscriber around their loan repayments, without pressure and
  without ever moving money:
    1. Reminder:  a calm heads-up before a debit.
    2. Failed:    a supportive message after a debit fails.
    3. Pause:     if the subscriber asks, it PROPOSES a pause and STOPS.
                  The subscriber confirms it themselves in the app.

How the file is laid out (read it top to bottom):
    SECTION 1  Settings and the form      (load the rulebooks)
    SECTION 2  The four tools             (how the Coach gets facts)
    SECTION 3  The answer check           (does the answer fit the form?)
    SECTION 4  The word check             (banned words, invented numbers)
    SECTION 5  The AI step                (the prompt and the model)
    SECTION 6  The three jobs             (reminder, failed, pause)
    SECTION 7  The main door: run(event)  (the Orchestrator calls this)
    SECTION 8  Self-check                 (run:  python coach.py)

The AI only writes the words (title and body). Amounts, dates and buttons
come from checked data and code, so the AI cannot invent them.
"""

import json
import re
from datetime import date
from pathlib import Path

import yaml                                            # pip install pyyaml
from jsonschema import Draft202012Validator, FormatChecker   # pip install jsonschema


HERE = Path(__file__).parent   # the folder this file sits in


# ===============================================================
# SECTION 1: SETTINGS AND THE FORM
# Load the two rulebooks one time:
#   output.schema.json  = the form every answer must fit
#   config.yaml         = allowed tools, message limits, banned phrases
# ===============================================================
SCHEMA = json.loads((HERE / "output.schema.json").read_text(encoding="utf-8"))
CONFIG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
BANNED_PHRASES = CONFIG["banned_phrases"]

ALLOWED_KINDS = ("reminder", "failed_help", "pause_proposal")

# The briefs for the AI live in ONE file, with one labelled section per job.
PROMPTS_FILE = "prompts_v1.md"
PROMPT_KINDS = ("reminder", "failed_help")

# Turn a reason code into plain words the subscriber can understand.
REASON_WORDS = {"insufficient_funds": "there was not enough money in the account"}
DEFAULT_REASON = "the bank did not complete the payment"

# The buttons the app shows. Our code owns these, never the AI.
SEE_SCHEDULE = {"label": "See my schedule", "action": "see_schedule"}
ASK_TO_PAUSE = {"label": "Ask to pause", "action": "ask_to_pause"}


class HandoffError(Exception):
    """The answer does not fit the form. Carries a list of problems."""
    def __init__(self, problems):
        self.problems = problems
        super().__init__("; ".join(problems))


class AgentError(Exception):
    """The Coach could not do its job safely."""
    pass


# ===============================================================
# SECTION 2: THE FOUR TOOLS
# These are the Coach's only "hands". None of them moves money.
# The data is FAKE for now. Later, each function is replaced by the
# real call. The names and the shape of the answers stay the same.
# ===============================================================
_FAKE_SCHEDULE = [
    {"collection_id": "col_104", "due_date": "2026-11-25", "amount_ngn": 20000},
    {"collection_id": "col_105", "due_date": "2026-12-25", "amount_ngn": 20000},
]

_FAKE_COLLECTION_LOG = [
    {"collection_id": "col_102", "date": "2026-09-15", "amount_ngn": 20000,
     "status": "success", "reason_code": None},
    {"collection_id": "col_103", "date": "2026-10-15", "amount_ngn": 20000,
     "status": "failed", "reason_code": "insufficient_funds"},
]

_DRAFTS = []   # saved drafts (kept in memory for testing)


def get_schedule(subscriber_ref):
    """READ: the upcoming debits. subscriber_ref is a safe token, never a real name."""
    return {"subscriber_ref": subscriber_ref, "schedule": list(_FAKE_SCHEDULE)}


def get_collection_log(subscriber_ref):
    """READ: what happened to past debits. This is how the Coach checks facts."""
    return {"subscriber_ref": subscriber_ref, "log": list(_FAKE_COLLECTION_LOG)}


def propose_pause(subscriber_ref, mandate_ref, pause_from, pause_until):
    """PROPOSE a pause. It does NOT pause anything. It always answers
    'input_required', which means a person must confirm first."""
    if date.fromisoformat(pause_until) <= date.fromisoformat(pause_from):
        raise ValueError("pause_until must be after pause_from")
    return {"status": "input_required", "subscriber_ref": subscriber_ref,
            "mandate_ref": mandate_ref, "pause_from": pause_from, "pause_until": pause_until}


def create_draft(subscriber_ref, title, body):
    """SAVE a message as a draft. Nothing is sent."""
    draft_id = f"draft_{len(_DRAFTS) + 1}"
    _DRAFTS.append({"draft_id": draft_id, "subscriber_ref": subscriber_ref,
                    "title": title, "body": body})
    return {"draft_id": draft_id, "saved": True}


# The phonebook: tool name -> the function that does the job.
TOOLS = {
    "repayment.get_schedule": get_schedule,
    "repayment.get_collection_log": get_collection_log,
    "repayment.propose_pause": propose_pause,
    "recommendation.create_draft": create_draft,
}


def call_tool(name, **arguments):
    """The receptionist. Connects the call only if the name is in the phonebook.
    (The REAL permission check belongs to the policy engine. This is a safety net.)"""
    if name not in TOOLS:
        raise PermissionError(f"Tool not allowed for the Repayment Coach: {name}")
    return TOOLS[name](**arguments)


# ===============================================================
# SECTION 3: THE ANSWER CHECK  (the inspector)
# Checks a finished answer against the form in output.schema.json.
# ===============================================================
def _make_checker(kind):
    """Build a checker for one kind of answer. Picking by kind gives clear
    messages like 'amount_ngn is wrong' instead of a vague 'no form matches'."""
    one_kind_schema = {
        "$schema": SCHEMA["$schema"],
        "$defs": SCHEMA["$defs"],             # the shared building blocks
        "$ref": f"#/$defs/{kind}",            # 'use the rules for this kind'
    }
    # FormatChecker makes sure dates really look like dates (2026-10-15).
    return Draft202012Validator(one_kind_schema, format_checker=FormatChecker())


_CHECKERS = {kind: _make_checker(kind) for kind in ALLOWED_KINDS}


def check_answer(answer):
    """Return the answer if it fits the form. Raise HandoffError if not."""
    if not isinstance(answer, dict):
        raise HandoffError(["answer must be a dictionary"])

    kind = answer.get("kind")
    if kind not in ALLOWED_KINDS:
        raise HandoffError([f"kind must be one of {ALLOWED_KINDS}, got {kind!r}"])

    problems = []   # collect every problem, so we can report them all at once
    for error in _CHECKERS[kind].iter_errors(answer):
        where = ".".join(str(part) for part in error.absolute_path) or "(whole answer)"
        problems.append(f"{where}: {error.message}")

    if problems:
        raise HandoffError(problems)
    return answer


def hand_over(answer):
    """Check the answer, then turn it into JSON text ready to pass on.
    (For the Orchestrator. The Coach itself calls check_answer.)"""
    return json.dumps(check_answer(answer), ensure_ascii=False)


# ===============================================================
# SECTION 4: THE WORD CHECK
# Reads the message and checks two things:
#   1. no banned phrase (threats, shaming, penalties, ...)
#   2. every number comes from the facts (the AI may not invent any)
# The Orchestrator does the official version. This is a second lock.
# ===============================================================
def banned_phrases_in(text):
    """Return the banned phrases found in the text (empty list = clean)."""
    found = []
    for phrase in BANNED_PHRASES:
        # \b means 'whole word only', so 'fee' does not match 'feel'.
        if re.search(r"\b" + re.escape(phrase) + r"\b", text, flags=re.IGNORECASE):
            found.append(phrase)
    return found


def numbers_in(text):
    """Pull every number out of the text. '₦20,000' becomes 20000."""
    return {int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\d|\d", text)}


def allowed_numbers(answer):
    """The only numbers a message may contain: the amount, and the day and
    year of each date in the answer (for example 25 and 2026)."""
    allowed = set()
    if "amount_ngn" in answer:
        allowed.add(int(answer["amount_ngn"]))
    for key in ("due_date", "failed_date", "pause_from", "pause_until"):
        if key in answer:
            d = date.fromisoformat(answer[key])
            allowed.update({d.day, d.year})
    return allowed


def message_problems(answer):
    """Check the message in a finished answer. An empty list means it is fine."""
    text = answer["message"]["title"] + " " + answer["message"]["body"]
    problems = []

    for phrase in banned_phrases_in(text):
        problems.append(f"banned phrase: {phrase!r}")

    not_in_facts = numbers_in(text) - allowed_numbers(answer)
    if not_in_facts:
        problems.append(f"numbers that are not in the facts: {sorted(not_in_facts)}")

    # A reminder or failed-payment message must say the amount.
    if "amount_ngn" in answer and int(answer["amount_ngn"]) not in numbers_in(text):
        problems.append("the message does not mention the amount")

    return problems


# ===============================================================
# SECTION 5: THE AI STEP
# A "model" is just a function: it takes the prompt text and returns the
# reply text. RIGHT NOW we use a FAKE one so we can build and test without
# a real AI. The real model (through the Model gateway) plugs in at the
# same spot, and nothing else in this file changes.
# ===============================================================
def nice_date(iso_text):
    """'2026-11-25' -> '25 November 2026'"""
    d = date.fromisoformat(iso_text)
    return f"{d.day} {d.strftime('%B %Y')}"


def load_prompt(kind):
    """Read prompts_v1.md and return the section for this kind of job.
    Each section starts with a marker line such as:  === PROMPT: reminder ==="""
    text = (HERE / PROMPTS_FILE).read_text(encoding="utf-8")
    # re.split gives: [text before, name1, body1, name2, body2, ...]
    parts = re.split(r"^=== PROMPT: (\w+) ===[ \t]*$", text, flags=re.MULTILINE)
    sections = dict(zip(parts[1::2], parts[2::2]))
    if kind not in sections:
        raise AgentError(f"{PROMPTS_FILE} has no section '=== PROMPT: {kind} ==='")
    return sections[kind]


def fill_prompt(kind, facts):
    """Take the prompt for this job and replace each <<name>> with its real fact."""
    text = load_prompt(kind)
    for name, value in facts.items():
        spot = f"<<{name}>>"
        # A missing spot means the prompt file is damaged (for example, copied
        # from a chat window). Say so clearly instead of guessing.
        if spot not in text:
            raise AgentError(f"The '{kind}' prompt in {PROMPTS_FILE} is missing the spot "
                             f"{spot}. Download the file again instead of copying and pasting.")
        text = text.replace(spot, str(value))
    if "<<" in text:
        raise AgentError("A fact is missing from the prompt")
    return text


def fake_model(prompt):
    """The pretend AI. It reads the facts from the prompt and writes a fixed-style message."""
    amount_found = re.search(r"Amount: (\d+) naira", prompt)
    date_found = re.search(r"Debit date: (.+)", prompt)
    if amount_found is None or date_found is None:
        raise AgentError("The prompt has no 'Amount' or 'Debit date' line. "
                         "Check that prompts_v1.md is complete.")
    amount = int(amount_found.group(1))
    date_text = date_found.group(1).strip()

    if "Reason, in plain words:" in prompt:       # the failed-payment prompt
        reason = re.search(r"Reason, in plain words: (.+)", prompt).group(1).strip()
        reply = {
            "title": "Payment not completed",
            "body": (f"Your ₦{amount:,} repayment on {date_text} did not go through "
                     f"because {reason}. You can fund your account, "
                     f"or ask to pause collections."),
        }
    else:                                         # the reminder prompt
        reply = {
            "title": "Repayment coming up",
            "body": (f"A repayment of ₦{amount:,} is due on {date_text}. "
                     f"Please make sure your account is ready."),
        }
    return json.dumps(reply, ensure_ascii=False)


def read_model_reply(reply_text):
    """Turn the model's reply text into {'title':..., 'body':...}."""
    text = reply_text.strip()
    # Some models wrap JSON in ```json ... ```. Remove that wrapper.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        reply = json.loads(text)
    except json.JSONDecodeError:
        raise AgentError("The model did not reply with valid JSON")
    if not isinstance(reply, dict) or not isinstance(reply.get("title"), str) \
            or not isinstance(reply.get("body"), str):
        raise AgentError("The model reply must have a text title and a text body")
    # Keep ONLY title and body. Anything else the model adds is thrown away.
    return {"title": reply["title"], "body": reply["body"]}


# ===============================================================
# SECTION 6: THE THREE JOBS
# ===============================================================
def make_reminder(event, model):
    """JOB 1: a payment is coming up -> reminder."""
    # Check the facts ourselves: look the debit up in the schedule.
    schedule = call_tool("repayment.get_schedule",
                         subscriber_ref=event["subscriber_ref"])["schedule"]
    debit = next((row for row in schedule
                  if row["collection_id"] == event.get("collection_id")), None)
    if debit is None:
        raise AgentError("That debit is not in the subscriber's schedule")

    prompt = fill_prompt("reminder", {
        "amount_ngn": debit["amount_ngn"],
        "due_date": nice_date(debit["due_date"]),
    })
    message = read_model_reply(model(prompt))

    return {
        "schema_version": "1.0",
        "kind": "reminder",
        "run_id": event["run_id"],
        "subscriber_ref": event["subscriber_ref"],
        "amount_ngn": debit["amount_ngn"],          # from checked data, not the model
        "due_date": debit["due_date"],               # from checked data, not the model
        "message": message,
        "buttons": [SEE_SCHEDULE],
    }


def make_failed_help(event, model):
    """JOB 2: a payment failed -> supportive help message."""
    # Do NOT trust the event. Ask the collection log what really happened.
    log = call_tool("repayment.get_collection_log",
                    subscriber_ref=event["subscriber_ref"])["log"]
    entry = next((row for row in log
                  if row["collection_id"] == event.get("collection_id")), None)
    if entry is None:
        raise AgentError("That debit is not in the collection log")
    if entry["status"] != "failed":
        raise AgentError("The log says this debit did not fail, so no failure message")

    reason_code = entry["reason_code"] or "unknown"
    prompt = fill_prompt("failed_help", {
        "amount_ngn": entry["amount_ngn"],
        "failed_date": nice_date(entry["date"]),
        "reason_plain": REASON_WORDS.get(reason_code, DEFAULT_REASON),
    })
    message = read_model_reply(model(prompt))

    return {
        "schema_version": "1.0",
        "kind": "failed_help",
        "run_id": event["run_id"],
        "subscriber_ref": event["subscriber_ref"],
        "amount_ngn": entry["amount_ngn"],
        "failed_date": entry["date"],
        "reason_code": reason_code,
        "message": message,
        "buttons": [SEE_SCHEDULE, ASK_TO_PAUSE],
    }


def make_pause_proposal(event, model):
    """JOB 3: the subscriber asked to pause -> PROPOSE it, then STOP.
    The message is fixed text, not written by the model, because it is
    about money terms. The model is never used here."""
    result = call_tool("repayment.propose_pause",
                       subscriber_ref=event["subscriber_ref"],
                       mandate_ref=event["mandate_ref"],
                       pause_from=event["pause_from"],
                       pause_until=event["pause_until"])

    # THE KEY SAFETY LINE: if the answer is not 'input_required', stop.
    if result["status"] != "input_required":
        raise AgentError("Unexpected pause status. The Coach stops here.")

    body = (f"You asked to pause collections from {nice_date(result['pause_from'])} "
            f"to {nice_date(result['pause_until'])}. "
            f"Please check the details and confirm in the app.")
    return {
        "schema_version": "1.0",
        "kind": "pause_proposal",
        "run_id": event["run_id"],
        "subscriber_ref": event["subscriber_ref"],
        "status": "input_required",     # nothing has changed. A person must confirm.
        "mandate_ref": result["mandate_ref"],
        "pause_from": result["pause_from"],
        "pause_until": result["pause_until"],
        "message": {"title": "Pause request", "body": body},
    }


# Which job handles which event (the switchboard).
JOBS = {
    "recova.collection.notification": make_reminder,
    "recova.collection.failed": make_failed_help,
    "pause.requested": make_pause_proposal,
}


# ===============================================================
# SECTION 7: THE MAIN DOOR
# The Orchestrator calls this one function.
# ===============================================================
def run(event, model=fake_model):
    job = JOBS.get(event.get("event_type"))                  # 1. pick the job
    if job is None:
        raise AgentError(f"The Coach does not handle this event: {event.get('event_type')!r}")

    answer = job(event, model)                               # 2. do the job

    check_answer(answer)                                     # 3. does it fit the form?

    problems = message_problems(answer)                      # 4. banned words, wrong numbers?
    if problems:
        raise AgentError("The message broke the content rules: " + "; ".join(problems))

    if answer["kind"] in ("reminder", "failed_help"):        # 5. save a draft (not sent)
        call_tool("recommendation.create_draft",
                  subscriber_ref=answer["subscriber_ref"],
                  title=answer["message"]["title"],
                  body=answer["message"]["body"])

    return answer                                            # 6. hand the answer back


# ===============================================================
# SECTION 8: SELF-CHECK
# Runs ONLY when you run this file directly:   python coach.py
# It does NOT run when another script imports coach.py.
# ===============================================================
def _self_test():
    checks = []   # (name, passed) pairs

    def check(name, condition):
        checks.append((name, bool(condition)))

    def raises(error_type, function):
        try:
            function()
            return False
        except error_type:
            return True

    # ---------- Example answers and events ----------
    good_failed = {
        "schema_version": "1.0", "kind": "failed_help", "run_id": "run_001",
        "subscriber_ref": "tok_abc123", "amount_ngn": 20000, "failed_date": "2026-10-15",
        "reason_code": "insufficient_funds",
        "message": {"title": "Payment not completed",
                    "body": "Your N20,000 repayment on 15 Oct was declined for insufficient funds."},
        "buttons": [SEE_SCHEDULE, ASK_TO_PAUSE],
    }
    good_pause = {
        "schema_version": "1.0", "kind": "pause_proposal", "run_id": "run_002",
        "subscriber_ref": "tok_abc123", "status": "input_required", "mandate_ref": "mandate_77",
        "pause_from": "2026-11-20", "pause_until": "2026-12-20",
        "message": {"title": "Pause request", "body": "Please confirm this pause."},
    }
    reminder_event = {"event_type": "recova.collection.notification", "run_id": "run_1",
                      "subscriber_ref": "tok_abc123", "collection_id": "col_104"}
    failed_event = {"event_type": "recova.collection.failed", "run_id": "run_2",
                    "subscriber_ref": "tok_abc123", "collection_id": "col_103"}
    pause_event = {"event_type": "pause.requested", "run_id": "run_3",
                   "subscriber_ref": "tok_abc123", "mandate_ref": "mandate_77",
                   "pause_from": "2026-11-20", "pause_until": "2026-12-20"}

    # ---------- Section 3: the answer check ----------
    check("form: good failed answer passes", check_answer(good_failed) == good_failed)
    check("form: good pause answer passes", check_answer(good_pause) == good_pause)
    check("form: extra field rejected", raises(HandoffError, lambda: check_answer({**good_failed, "secret": "hi"})))
    check("form: forbidden button rejected", raises(HandoffError, lambda: check_answer(
        {**good_failed, "buttons": [{"label": "Cancel loan", "action": "cancel_mandate"}]})))
    check("form: pause cannot claim 'done'", raises(HandoffError, lambda: check_answer({**good_pause, "status": "done"})))
    check("form: negative amount rejected", raises(HandoffError, lambda: check_answer({**good_failed, "amount_ngn": -5})))
    check("form: bad date rejected", raises(HandoffError, lambda: check_answer({**good_failed, "failed_date": "15th October"})))
    check("form: too-long message rejected", raises(HandoffError, lambda: check_answer(
        {**good_failed, "message": {"title": "Hi", "body": "x" * 400}})))
    check("form: unknown kind rejected", raises(HandoffError, lambda: check_answer({**good_failed, "kind": "cancel_everything"})))
    missing = dict(good_failed)
    del missing["amount_ngn"]
    check("form: missing field rejected", raises(HandoffError, lambda: check_answer(missing)))
    check("form: hand_over returns JSON text", '"kind": "failed_help"' in hand_over(good_failed))

    # ---------- Section 2: the tools ----------
    check("tools: schedule has 2 upcoming debits",
          len(call_tool("repayment.get_schedule", subscriber_ref="tok_abc123")["schedule"]) == 2)
    log = call_tool("repayment.get_collection_log", subscriber_ref="tok_abc123")["log"]
    failed_rows = [row for row in log if row["status"] == "failed"]
    check("tools: log shows the failed debit", len(failed_rows) == 1)
    check("tools: failed reason is insufficient_funds", failed_rows[0]["reason_code"] == "insufficient_funds")
    check("tools: pause answers input_required", call_tool(
        "repayment.propose_pause", subscriber_ref="tok_abc123", mandate_ref="m",
        pause_from="2026-11-20", pause_until="2026-12-20")["status"] == "input_required")
    check("tools: backwards pause dates refused", raises(ValueError, lambda: call_tool(
        "repayment.propose_pause", subscriber_ref="tok_abc123", mandate_ref="m",
        pause_from="2026-12-20", pause_until="2026-11-20")))
    draft = call_tool("recommendation.create_draft", subscriber_ref="tok_abc123",
                      title="Payment not completed", body="Your repayment was declined.")
    check("tools: draft saved with an id", draft["saved"] and draft["draft_id"].startswith("draft_"))
    check("tools: unknown tool refused", raises(PermissionError, lambda: call_tool("repayment.cancel_mandate")))
    check("tools: exactly four tools", len(TOOLS) == 4)

    # ---------- Section 4: the word check ----------
    def make(body):
        return {"amount_ngn": 20000, "due_date": "2026-11-25",
                "message": {"title": "Repayment coming up", "body": body}}

    check("words: clean message has no problems",
          message_problems(make("A repayment of ₦20,000 is due on 25 November 2026.")) == [])
    check("words: catches a threat", any("banned" in p for p in message_problems(
        make("Pay now or face legal action. ₦20,000, 25 November 2026."))))
    check("words: catches a wrong amount", any("not in the facts" in p for p in message_problems(
        make("A repayment of ₦50,000 is due on 25 November 2026."))))
    check("words: catches a wrong date", any("not in the facts" in p for p in message_problems(
        make("A repayment of ₦20,000 is due on 30 November 2026."))))
    check("words: catches a missing amount", any("mention the amount" in p for p in message_problems(
        make("A repayment is due on 25 November 2026."))))
    check("words: whole-word match only ('feel' is fine)",
          message_problems(make("We hope you feel ready. ₦20,000 is due on 25 November 2026.")) == [])

    # ---------- Sections 5 to 7: the jobs and run() ----------
    reminder = run(reminder_event)
    check("run: reminder has the right kind", reminder["kind"] == "reminder")
    check("run: reminder amount and date come from checked data",
          reminder["amount_ngn"] == 20000 and reminder["due_date"] == "2026-11-25")
    check("run: reminder message mentions the amount", "₦20,000" in reminder["message"]["body"])

    failed = run(failed_event)
    check("run: failed has the right kind and reason",
          failed["kind"] == "failed_help" and failed["reason_code"] == "insufficient_funds")
    check("run: failed has both buttons", len(failed["buttons"]) == 2)
    check("run: failed reason in plain words", "not enough money" in failed["message"]["body"])

    pause = run(pause_event)
    check("run: pause stops at input_required",
          pause["kind"] == "pause_proposal" and pause["status"] == "input_required")

    check("run: refuses a 'failed' event when the log says success",
          raises(AgentError, lambda: run({**failed_event, "collection_id": "col_102"})))
    check("run: refuses an unknown event type",
          raises(AgentError, lambda: run({**reminder_event, "event_type": "cancel.everything"})))
    check("run: refuses a model reply that is not JSON",
          raises(AgentError, lambda: run(reminder_event, model=lambda p: "hello there")))
    long_reply = json.dumps({"title": "Hi", "body": "x" * 400})
    check("run: the form check stops a too-long model message",
          raises(HandoffError, lambda: run(reminder_event, model=lambda p: long_reply)))
    sneaky = json.dumps({"title": "Hi", "body": "A repayment of ₦20,000 is due on 25 November 2026.",
                         "buttons": [{"label": "Cancel loan", "action": "cancel_mandate"}]})
    check("run: extra fields from the model are ignored",
          run(reminder_event, model=lambda p: sneaky)["buttons"] == [SEE_SCHEDULE])
    check("run: catches a prompt file with a missing spot", raises(AgentError, lambda: fill_prompt(
        "reminder", {"amount_ngn": 1, "due_date": "x", "not_in_file": "y"})))

    # ---------- Print the results ----------
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    print(f"\n{sum(ok for _, ok in checks)} of {len(checks)} checks passed")

    # Show one real answer so you can see what the Coach produces.
    print("\nExample failed-payment answer:")
    print(json.dumps(failed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _self_test()