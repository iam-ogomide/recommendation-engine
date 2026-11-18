# Repayment Coach: handover note

## What this is
The Repayment Coach node of the CreditCliq chatbot. It supports a subscriber around
repayments without pressure and without moving money. It has three jobs:

1. **Reminder** before a debit (`recova.collection.notification`).
2. **Help after a failed debit** (`recova.collection.failed`).
3. **Propose a pause** when the subscriber asks. The Coach only proposes and stops at
   `input_required`. A person confirms in the app.

Autonomy is A1, with a ceiling of A3, as in `SKILL.md`.

## How it works, in one paragraph
`agent.py` receives an event and checks the facts itself through the tools in `tool.py`
(it does not trust the event). For a reminder or failed payment it fills a prompt file,
asks the model for a title and body only, and builds the full answer around those words.
The amount, dates and buttons always come from checked data and code, never from the model.
For a pause, the model is not used at all. The answer is checked by `handoff.py` (the
form) and `rules.py` (banned phrases, numbers must match the facts).

## What is real and what is a stand-in
| Part | Status |
|---|---|
| Output schema, handoff check, content rules, agent logic | Real, tested |
| `tool.py` (four tools) | **Stand-in.** Returns fake data. Replace the inside of each function with the real MCP call. Names and answer shapes stay the same. |
| `fake_model` in `agent.py` | **Stand-in.** Replace with a call through the Model gateway. It must take the prompt text and return the reply text. |
| Prompt files `reminder_v1.md`, `failed_v1.md` | Written, **not yet tested against a real model** |

## Files
| File | Purpose |
|---|---|
| `SKILL.md` | The official job description, copied from the architecture file |
| `output.schema.json` | The only answer shapes the Coach may hand over |
| `handoff.py` | Checks an answer against the schema before it leaves |
| `tool.py` | The four allowed tools (fake data for now) |
| `agent.py` | The main logic. `run(event)` is the one entry point. |
| `rules.py` | Banned phrases and number check |
| `reminder_v1.md`, `failed_v1.md` | Prompt files for the model |
| `tools.yaml`, `limits.yaml` | Allowed tools, message limits, banned phrase list |
| `evals.py` | Golden, red-team and config checks |

## How to run
```
pip install -r requirements.txt
python handoff.py
python tool.py
python rules.py
python agent.py
python evals.py
```
Expected: every script ends with "all checks passed" (34 of 34 for `evals.py`).

## Please confirm or decide (these are my assumptions)
1. **Event names and fields.** `recova.collection.notification` and
   `recova.collection.failed` come from the architecture file. `pause.requested` is my own
   name for the "Ask to pause" tap. Events are assumed to carry `event_type`, `run_id`,
   `subscriber_ref` and `collection_id`. A pause event also carries `mandate_ref`,
   `pause_from` and `pause_until`. Please confirm the real fields, and where the pause
   dates come from.
2. **Schema field names** (for example `reason_code`, `mandate_ref`, `amount_ngn`) are
   proposals. Confirm they match what the Orchestrator and RecovaPRO expect.
3. **Buttons.** A reminder shows one button (See my schedule). A failure message shows two.
   This was my choice.
4. **Reason codes.** Only `insufficient_funds` has plain-words text. Anything else falls
   back to "the bank did not complete the payment". Please supply the real RecovaPRO codes.
5. **Banned phrases** in `limits.yaml` are my proposal. The product team should review them.
6. **Tone and mandate references.** `SKILL.md` links to `references/mandate-states.md` and
   `references/tone.md`. They are not included.
7. **The model.** The architecture file proposes a small local model first. The Coach only
   needs one function that takes a prompt and returns text.

## What the Orchestrator still has to do
- Stop when the answer has `status: input_required`, show the exact terms in the app, and
  wait for the signed confirmation. The Coach never sees the confirmation.
- Keep its own official content check and the 08:00 to 20:00 WAT sending window. The
  content check inside the Coach is only a second lock.
- Enforce permissions through the policy engine. `tools.yaml` grants nothing by itself.

## Not included
`Dockerfile`, `CHANGELOG.md`, the `references/` files, and the folder layout from the
architecture file (`agents/repayment-coach/...`). This node is a flat folder for now.
Move the files into the standard layout when it joins the main repo.