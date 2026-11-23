"""CreditCliq Knowledge Base, in one file.  A platform component, not an agent.

Job: give the Concierge trustworthy guidance to cite, and nothing else.

    reviewed guidance (Markdown files)  ->  review gate  ->  embed  ->  store  ->  search (top 3 above a threshold)

  content/*.md   one file per item: owner, reviewer, review date and sources at the top, plain-language text below
  review gate    only items that pass are searchable (section 3)
  embeddings     turn text into numbers so a question can find guidance with the same meaning, not just the same words
  store          a local SQLite file for now; the interface is what pgvector will implement in production (see SCHEMA_PG)
  search         at most 3 items, each above a threshold chosen from data (--calibrate), with the item id to cite

What the review gate enforces (the knowledge-base SKILL.md)
  * Every item has an owner AND a different reviewer, a review date that is not in the future and not stale
    (over a year old drops out of search by itself), a status, and at least one source.
  * No bureau scoring methods, no lender names or promotions or offers, no promises, no fraud verdicts, no identifiers.
  * Plain language: short items, short sentences.
  * Sample items never reach search unless you ask for them (--include-samples). Real approval is compliance's job.
  * Retired or failing items are removed from search on the next load.

Embedders
  hash   a tiny word-based stand-in. No download. Good for tests and for seeing the flow; NOT for real quality.
  e5     multilingual-e5-small on ONNX Runtime, as the architecture says. Text stays on your machine. One-time setup:
           pip install onnxruntime tokenizers numpy
           python kb.py --download-model models/e5-small        (needs: pip install huggingface_hub)
           python kb.py --check-model models/e5-small           (tells you if a file is missing)
         then add  --embedder e5 --model-dir models/e5-small  to the commands below.
         I could not download the model where I built this, so the e5 code is tested with stand-ins for the model and
         tokenizer, not with the real files: run --eval with e5 and send me the result.

Run:
  python kb.py --selftest
  python kb.py --write-samples content          write the sample items as files (see the format)
  python kb.py --check content                  run the review gate on a folder; no model needed
  python kb.py --load content --include-samples [--embedder e5 --model-dir models/e5-small]
  python kb.py --search "what is an enquiry" [--embedder e5 --model-dir ...]
  python kb.py --eval  [--embedder e5 ...]      do the right items come back? do unrelated questions get nothing?
  python kb.py --calibrate [--embedder e5 ...]  suggest a threshold from the eval questions
  python kb.py --demo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import struct
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Protocol

from guardrails import LENDER_RE, has_pii_shape, scan_wording

# =============================================================================
# 1. SETTINGS  (edit here)
# =============================================================================

TOP_K = 3
REVIEW_MAX_AGE_DAYS = 365          # a review older than this drops the item out of search until it is reviewed again
TITLE_MAX = 80
BODY_MAX = 700                     # characters: a snippet must fit inside a Concierge answer
MAX_AVG_SENTENCE_WORDS = 28        # plain language
QUERY_MAX_CHARS = 300
STATUSES = ("draft", "approved", "retired")

# Search thresholds are PER EMBEDDER because each model scores similarity on its own scale.
# The e5 value is a PLACEHOLDER: e5 scores cluster high, so choose it from your data with --calibrate.
EMBEDDERS = {
    "hash": {"threshold": 0.30},
    "e5": {"threshold": 0.82},
}

FIELDS = {"id", "title", "topics", "owner", "reviewer", "reviewed_on", "status", "sources", "language", "sample"}

KB_BANNED_PATTERNS = {
    "BUREAU_METHOD": (
        r"\balgorithm\b", r"\bformula\b", r"\bscoring\s+(?:model|method|criteria)\b", r"\bweight(?:s|ed|ing)?\b",
        r"\bcalculat(?:e|es|ed|ion)\b", r"\bhow\s+(?:the\s+)?bureaus?\s+(?:score|rate)\b",
    ),
    "OFFER": (
        r"\bapply now\b", r"\blimited[- ]time\b", r"\bspecial offer\b", r"\bpre-?approved\b", r"\bpromo(?:tion|tional)?\b",
        r"\bdiscount\b", r"\bcashback\b", r"\binterest rate of\b", r"\b\d+(?:\.\d+)?\s*%\s*(?:interest|apr)\b",
        r"\bget (?:a|your) loan\b", r"\bloan offers?\b", r"\bsign up (?:today|now)\b", r"\bclick here\b",
    ),
    "PROMISE": (
        r"\bguarantee(?:d|s)?\b", r"\bwill\s+(?:increase|improve|raise|boost|fix)\s+your\s+score\b",
        r"\bwithin\s+\d+\s+(?:days|weeks|months)\b",
    ),
    "FRAUD_LABEL": (
        r"\bthis is fraud\b", r"\byou (?:have been|were) (?:defrauded|scammed)\b", r"\bidentity theft has\b",
    ),
}

SCHEMA_PG = """\
-- production store (PostgreSQL + pgvector): the same columns as the SQLite table; vector size 384 for multilingual-e5-small
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS kb_items (
  id            text PRIMARY KEY,
  title         text NOT NULL,
  body          text NOT NULL,
  topics        text[] NOT NULL,
  language      text NOT NULL DEFAULT 'en',
  owner         text NOT NULL,
  reviewer      text NOT NULL,
  reviewed_on   date NOT NULL,
  sources       text[] NOT NULL,
  content_hash  text NOT NULL,
  embedder_id   text NOT NULL,
  embedding     vector(384) NOT NULL,
  loaded_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS kb_items_embedding ON kb_items USING hnsw (embedding vector_cosine_ops);
-- search: SELECT id, title, body, 1 - (embedding <=> $1) AS score FROM kb_items
--         WHERE reviewed_on >= current_date - 365 ORDER BY embedding <=> $1 LIMIT 3;  then apply the threshold.
"""


# =============================================================================
# 2. CONTENT: files -> items
# =============================================================================

@dataclass
class Problem:
    code: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.code}" + (f": {self.detail}" if self.detail else "")


@dataclass
class Item:
    id: str = ""
    title: str = ""
    body: str = ""
    topics: list[str] = field(default_factory=list)
    owner: str = ""
    reviewer: str = ""
    reviewed_on: date | None = None
    status: str = ""
    sources: list[str] = field(default_factory=list)
    language: str = "en"
    sample: bool = False
    path: str = ""

    def content_hash(self) -> str:
        blob = json.dumps([self.title, self.body, sorted(self.topics), self.language], sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _unquote(v: str) -> str:
    v = v.strip()
    return v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'" else v


def parse_item(text: str, path: str = "") -> tuple[Item, list[Problem]]:
    """Parse one Markdown file. Returns the item (possibly incomplete) and any problems found while parsing."""
    problems: list[Problem] = []
    text = text.replace("\r\n", "\n").lstrip("\ufeff")
    item = Item(path=path)
    if not text.startswith("---\n"):
        return item, [Problem("NO_FRONT_MATTER", "the file must start with a --- block holding owner, reviewer and dates")]
    end = text.find("\n---", 4)
    if end == -1:
        return item, [Problem("NO_FRONT_MATTER", "the --- block is never closed")]
    head, body = text[4:end], text[end + 4:].strip("\n")
    item.body = re.sub(r"[ \t]+\n", "\n", body).strip()
    values: dict[str, Any] = {}
    key: str | None = None
    for raw in head.split("\n"):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_]+):\s*(.*)$", raw)
        if m and not raw.startswith((" ", "\t")):
            key, val = m.group(1), m.group(2).strip()
            if key not in FIELDS:
                problems.append(Problem("UNKNOWN_FIELD", key))
                key = None
                continue
            if val == "":
                values[key] = []   # a block list follows
            elif val.startswith("[") and val.endswith("]"):
                values[key] = [_unquote(x) for x in val[1:-1].split(",") if x.strip()]
            else:
                values[key] = _unquote(val)
        elif key and re.match(r"^\s+-\s+", raw) and isinstance(values.get(key), list):
            values[key].append(_unquote(re.sub(r"^\s+-\s+", "", raw)))
        else:
            problems.append(Problem("BAD_LINE", raw.strip()[:40]))
    item.id = str(values.get("id", ""))
    item.title = str(values.get("title", ""))
    item.owner = str(values.get("owner", ""))
    item.reviewer = str(values.get("reviewer", ""))
    item.status = str(values.get("status", "")).lower()
    item.language = str(values.get("language", "en")) or "en"
    item.sample = str(values.get("sample", "false")).lower() == "true"
    for name in ("topics", "sources"):
        v = values.get(name, [])
        setattr(item, name, v if isinstance(v, list) else [v])
    if "reviewed_on" in values:
        try:
            item.reviewed_on = date.fromisoformat(str(values["reviewed_on"]))
        except ValueError:
            problems.append(Problem("BAD_DATE", "reviewed_on must look like 2026-11-01"))
    return item, problems


def read_folder(folder: str) -> list[tuple[Item, list[Problem]]]:
    out = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".md"):
            p = os.path.join(folder, name)
            with open(p, encoding="utf-8") as fh:
                out.append(parse_item(fh.read(), p))
    return out


# =============================================================================
# 3. THE REVIEW GATE  (review.py)
# =============================================================================

def review_item(item: Item, today: date, allow_samples: bool = False) -> list[Problem]:
    """Every reason this item may not be searchable. An empty list means it may."""
    p: list[Problem] = []
    if not re.match(r"^[a-z0-9][a-z0-9\-]{1,60}$", item.id):
        p.append(Problem("BAD_ID", "use lower-case letters, digits and hyphens, e.g. pay-on-time"))
    if not item.title.strip():
        p.append(Problem("NO_TITLE"))
    elif len(item.title) > TITLE_MAX:
        p.append(Problem("TITLE_TOO_LONG", f"{len(item.title)} > {TITLE_MAX}"))
    if not item.body.strip():
        p.append(Problem("NO_BODY"))
    elif len(item.body) > BODY_MAX:
        p.append(Problem("BODY_TOO_LONG", f"{len(item.body)} > {BODY_MAX} characters"))
    if not item.topics:
        p.append(Problem("NO_TOPICS"))
    if item.status not in STATUSES:
        p.append(Problem("BAD_STATUS", f"must be one of {STATUSES}"))
    if item.status == "approved":
        if not item.owner.strip():
            p.append(Problem("NO_OWNER"))
        if not item.reviewer.strip():
            p.append(Problem("NO_REVIEWER"))
        elif item.reviewer.strip().lower() == item.owner.strip().lower():
            p.append(Problem("REVIEWER_IS_OWNER", "a different person must review it"))
        if item.reviewed_on is None:
            p.append(Problem("NO_REVIEW_DATE"))
        elif item.reviewed_on > today:
            p.append(Problem("REVIEW_DATE_IN_FUTURE"))
        elif (today - item.reviewed_on).days > REVIEW_MAX_AGE_DAYS:
            p.append(Problem("STALE_REVIEW", f"reviewed {(today - item.reviewed_on).days} days ago; review again"))
        if not [s for s in item.sources if s.strip()]:
            p.append(Problem("NO_SOURCE", "name at least one source"))
    if item.sample and not allow_samples:
        p.append(Problem("SAMPLE_CONTENT", "a sample is not real approved guidance"))

    text = f"{item.title}. {item.body}"
    for code, matched in scan_wording(text, patterns=KB_BANNED_PATTERNS, exempt=()):
        p.append(Problem(code, f"matched {matched!r}"))
    m = LENDER_RE.search(text)
    if m:
        p.append(Problem("LENDER", f"names a lender or bank ({m.group(0)!r})"))
    if has_pii_shape(text):
        p.append(Problem("PII", "looks like a BVN, account number, phone or email"))
    sentences = [s for s in re.split(r"[.!?]+\s*", item.body) if s.strip()]
    if sentences:
        avg = sum(len(s.split()) for s in sentences) / len(sentences)
        if avg > MAX_AVG_SENTENCE_WORDS:
            p.append(Problem("NOT_PLAIN", f"sentences average {avg:.0f} words; keep them under {MAX_AVG_SENTENCE_WORDS}"))
    return p


# =============================================================================
# 4. EMBEDDERS
# =============================================================================

class Embedder(Protocol):
    id: str

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


def _norm(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v


_STOP = set("a an the of to in on for and or is are was were be been it its this that these those you your yours i my me we our "
            "do does did can could should would will with as at by from if so than then there what which who whom how when why "
            "about into not no any some".split())


def _stem(w: str) -> str:
    for suf in ("ments", "ment", "ings", "ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


class HashEmbedder:
    """A tiny word-based stand-in so everything can be tested without downloading a model. Shared words (after a
    crude stem) give similarity; it knows nothing about meaning. Do not use it to judge real quality."""

    id = "hash-256-v1"
    DIM = 256

    def _one(self, text: str) -> list[float]:
        words = [_stem(w) for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP]
        v = [0.0] * self.DIM
        for w in words:
            h = int(hashlib.blake2b(w.encode(), digest_size=8).hexdigest(), 16)
            v[h % self.DIM] += 1.0
        return _norm(v)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class E5Embedder:
    """multilingual-e5-small on ONNX Runtime. Runs locally; text never leaves the machine.

    e5 needs 'query: ' in front of questions and 'passage: ' in front of documents, mean pooling over the real
    (non-padding) tokens, and unit-length vectors."""

    id = "multilingual-e5-small"
    MAX_TOKENS = 512
    BATCH = 16

    def __init__(self, model_dir: str, quantized: bool = False, threads: int | None = None):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("the e5 embedder needs: pip install onnxruntime tokenizers numpy") from exc
        onnx_dir = os.path.join(model_dir, "onnx")
        names = ["model_quantized.onnx", "model.onnx"] if quantized else ["model.onnx", "model_quantized.onnx"]
        model_path = next((os.path.join(d, n) for d in (onnx_dir, model_dir) for n in names if os.path.exists(os.path.join(d, n))), None)
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if model_path is None or not os.path.exists(tok_path):
            raise RuntimeError(f"model files not found in {model_dir}: run  python kb.py --check-model {model_dir}")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._setup(session, Tokenizer.from_file(tok_path))

    @classmethod
    def from_parts(cls, session: Any, tokenizer: Any) -> "E5Embedder":
        """Build from an already-created session and tokenizer (used by the tests with stand-ins)."""
        self = cls.__new__(cls)
        self._setup(session, tokenizer)
        return self

    def _setup(self, session: Any, tokenizer: Any) -> None:
        self._session, self._tok = session, tokenizer
        pad = tokenizer.token_to_id("<pad>") if hasattr(tokenizer, "token_to_id") else None
        tokenizer.enable_truncation(max_length=self.MAX_TOKENS)
        tokenizer.enable_padding(pad_id=pad if pad is not None else 1, pad_token="<pad>")
        self._inputs = [i.name for i in session.get_inputs()]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np
        out: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH):
            enc = self._tok.encode_batch(texts[start:start + self.BATCH])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds: dict[str, Any] = {}
            if "input_ids" in self._inputs:
                feeds["input_ids"] = ids
            if "attention_mask" in self._inputs:
                feeds["attention_mask"] = mask
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self._session.run(None, feeds)[0]
            if hidden.ndim == 3:   # [batch, tokens, size]: average the real tokens only
                m = mask[:, :, None].astype(hidden.dtype)
                pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            else:                  # the export already pooled it
                pooled = hidden
            pooled = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out.extend(pooled.astype(float).tolist())
        return out

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(["query: " + t for t in texts])

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._embed(["passage: " + t for t in texts])


def make_embedder(name: str, model_dir: str | None = None) -> Embedder:
    if name == "hash":
        return HashEmbedder()
    if name == "e5":
        if not model_dir:
            raise RuntimeError("--embedder e5 needs --model-dir (see --download-model and --check-model)")
        return E5Embedder(model_dir)
    raise ValueError(f"unknown embedder {name!r}")


def threshold_for(embedder: Embedder) -> float:
    return EMBEDDERS["e5" if isinstance(embedder, E5Embedder) or embedder.id.startswith("multilingual-e5") else "hash"]["threshold"]


# 5. THE STORE AND SEARCH  (ingest.py + search.py)

@dataclass
class Hit:
    id: str
    title: str
    body: str
    score: float
    topics: list[str]
    sources: list[str]


@dataclass
class LoadReport:
    loaded: list[str] = field(default_factory=list)       # embedded this time
    unchanged: list[str] = field(default_factory=list)    # already stored, same text: not re-embedded
    removed: list[str] = field(default_factory=list)      # no longer searchable (retired, stale, failing, deleted)
    rejected: dict[str, list[Problem]] = field(default_factory=dict)   # id or file -> why it is not searchable


def _pack(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def _unpack(b: bytes) -> list[float]:
    return list(struct.unpack(f"{len(b) // 4}f", b))


class KnowledgeBase:
    def __init__(self, path: str, embedder: Embedder, threshold: float | None = None,
                 today: Callable[[], date] = date.today):
        self._emb, self._today = embedder, today
        self.threshold = threshold if threshold is not None else threshold_for(embedder)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL, topics TEXT NOT NULL, "
            "language TEXT NOT NULL, owner TEXT NOT NULL, reviewer TEXT NOT NULL, reviewed_on TEXT NOT NULL, sources TEXT NOT NULL, "
            "content_hash TEXT NOT NULL, embedder_id TEXT NOT NULL, embedding BLOB NOT NULL)")
        self._db.commit()

    # ---- load (ingest) -----------------------------------------------------------
    def load(self, parsed: list[tuple[Item, list[Problem]]], allow_samples: bool = False) -> LoadReport:
        """Make the store match the content: embed what is new or changed, remove what is no longer allowed."""
        rep = LoadReport()
        today = self._today()
        good: dict[str, Item] = {}
        seen: dict[str, str] = {}
        for item, parse_problems in parsed:
            label = item.id or os.path.basename(item.path) or "(no id)"
            problems = list(parse_problems)
            if item.status == "retired":   # retired is a valid way to leave search: not a failure
                seen.setdefault(item.id, item.path)
                continue
            problems += review_item(item, today, allow_samples)
            if item.status == "draft":   # only APPROVED items are ever searchable
                problems.append(Problem("DRAFT", "not approved yet, so not searchable"))
            if item.id in seen:
                problems.append(Problem("DUPLICATE_ID", f"also in {seen[item.id]}"))
            elif item.id:
                seen[item.id] = item.path
            if problems:
                rep.rejected[label] = problems
            else:
                good[item.id] = item
        with self._lock:
            stored = {r[0]: (r[1], r[2]) for r in self._db.execute("SELECT id, content_hash, embedder_id FROM items")}
            todo = [i for i in good.values() if stored.get(i.id) != (i.content_hash(), self._emb.id)]
            rep.unchanged = [i for i in good if i not in {t.id for t in todo}]
            if todo:
                vecs = self._emb.embed_passages([f"{i.title}. {i.body}" for i in todo])
                for it, v in zip(todo, vecs):
                    self._db.execute(
                        "INSERT OR REPLACE INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (it.id, it.title, it.body, json.dumps(it.topics), it.language, it.owner, it.reviewer,
                         it.reviewed_on.isoformat() if it.reviewed_on else "", json.dumps(it.sources), it.content_hash(),
                         self._emb.id, _pack(v)))
                    rep.loaded.append(it.id)
            for gone in [k for k in stored if k not in good]:
                self._db.execute("DELETE FROM items WHERE id = ?", (gone,))
                rep.removed.append(gone)
            self._db.commit()
        return rep

    # ---- search --------------------------------------------------------------------
    def search(self, query: str, k: int = TOP_K, threshold: float | None = None, topics: list[str] | None = None,
               language: str | None = None) -> list[Hit]:
        """At most k items whose similarity clears the threshold, best first. Items whose review has gone stale since
        the last load are skipped even if still stored."""
        q = " ".join(query.split())[:QUERY_MAX_CHARS]
        if not q or has_pii_shape(q):   # an identifier is never embedded or searched
            return []
        k = min(max(k, 0), TOP_K)
        cut = self.threshold if threshold is None else threshold
        oldest = (self._today() - timedelta(days=REVIEW_MAX_AGE_DAYS)).isoformat()
        with self._lock:
            rows = self._db.execute(
                "SELECT id, title, body, topics, language, sources, reviewed_on, embedder_id, embedding FROM items").fetchall()
        if not rows:
            return []
        if any(r[7] != self._emb.id for r in rows):
            raise RuntimeError("the stored vectors came from a different embedder: load the content again with this one")
        qv = self._emb.embed_queries([q])[0]
        hits: list[Hit] = []
        for id_, title, body, tj, lang, sj, reviewed_on, _, blob in rows:
            if reviewed_on < oldest or (language and lang != language):
                continue
            tl = json.loads(tj)
            if topics and not set(topics) & set(tl):
                continue
            score = sum(a * b for a, b in zip(qv, _unpack(blob)))
            if score >= cut:
                hits.append(Hit(id_, title, body, round(score, 4), tl, json.loads(sj)))
        hits.sort(key=lambda h: (-h.score, h.id))
        return hits[:k]

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    def ids(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT id FROM items ORDER BY id")]

    def close(self) -> None:
        with self._lock:
            self._db.close()


class KnowledgeSearchTool:
    """The `knowledge.search` tool: plug it into the Concierge's tool belt.

        Concierge(llm, handlers={"knowledge.search": KnowledgeSearchTool(kb)})

    Returns {"items": [{"id", "title", "body"}, ...]} (at most 3), the shape the Concierge already expects."""

    def __init__(self, kb: KnowledgeBase, k: int = TOP_K):
        self._kb, self._k = kb, k

    def __call__(self, args: dict) -> dict:
        query = args.get("query") if isinstance(args, dict) else None
        hits = self._kb.search(query, k=self._k) if isinstance(query, str) else []
        return {"items": [{"id": h.id, "title": h.title, "body": h.body} for h in hits]}


# =============================================================================
# 6. EVALS: do the right items come back, and does an unrelated question get nothing?
# =============================================================================

# (question, ids of which AT LEAST ONE must come back).  An empty tuple means NOTHING should come back.
EVAL_CASES: list[tuple[str, tuple[str, ...]]] = [
    ("What is an enquiry on my report?", ("what-is-an-enquiry",)),
    ("Why is there an enquiry I do not remember?", ("what-is-an-enquiry", "unfamiliar-account")),
    ("There is an account on my report that is not mine", ("unfamiliar-account",)),
    ("What does debt burden ratio mean?", ("debt-burden-ratio",)),
    ("How can I avoid missing loan payments?", ("pay-on-time", "small-cash-buffer")),
    ("Someone is asking me for my OTP", ("protect-your-details",)),
    ("How do I read my credit report?", ("reading-your-report",)),
    ("Should I keep my payment receipts?", ("keep-payment-records",)),
    ("What is the weather in Lagos today?", ()),
    ("Recommend a good restaurant for dinner", ()),
    ("Tell me a joke about football", ()),
    ("Who won the match last night?", ()),
]


def run_eval(kb: KnowledgeBase, cases=EVAL_CASES, threshold: float | None = None) -> dict:
    pos = neg = recall_hits = false_accepts = 0
    rows = []
    for q, want in cases:
        got = kb.search(q, threshold=threshold)
        ids = [h.id for h in got]
        if want:
            pos += 1
            ok = bool(set(ids) & set(want))
            recall_hits += ok
        else:
            neg += 1
            ok = not ids
            false_accepts += not ok
        rows.append({"question": q, "want": list(want), "got": [(h.id, h.score) for h in got], "ok": ok})
    return {"recall": round(recall_hits / pos, 3) if pos else 1.0, "false_accept_rate": round(false_accepts / neg, 3) if neg else 0.0,
            "positives": pos, "negatives": neg, "rows": rows}


def calibrate(kb: KnowledgeBase, cases=EVAL_CASES) -> dict:
    """Best score of the RIGHT item for each positive question, and best score of ANYTHING for each negative one.
    A good threshold sits between the lowest right-answer score and the highest wrong-question score."""
    pos_scores, neg_scores = [], []
    for q, want in cases:
        got = kb.search(q, threshold=-1.0, k=TOP_K)
        if want:
            right = [h.score for h in got if h.id in want]
            pos_scores.append(max(right) if right else float("-inf"))
        else:
            neg_scores.append(max((h.score for h in got), default=0.0))
    lo, hi = min(pos_scores, default=0.0), max(neg_scores, default=0.0)
    return {"lowest_right_answer": round(lo, 4), "highest_unrelated": round(hi, 4),
            "separable": lo > hi, "suggested_threshold": round((lo + hi) / 2, 3) if lo > hi else None}


# =============================================================================
# 7. SAMPLE ITEMS  (examples of the format; NOT approved guidance)
# =============================================================================

_SAMPLE_SRC = "SAMPLE: replace with a real source before this is approved"
SAMPLE_ITEMS: list[dict[str, Any]] = [
    {"id": "pay-on-time", "title": "Paying on time", "topics": ["payments", "recommendations"],
     "body": "Paying your loans and credit facilities on or before the due date is one of the most practical things you can do "
             "to keep your report in good shape. If a payment is hard to make, contact your lender early to talk about options."},
    {"id": "what-is-an-enquiry", "title": "What is an enquiry on your report?", "topics": ["report", "enquiries"],
     "body": "An enquiry is a record that a lender or another organisation looked at your credit report, usually because you "
             "applied for credit. Check that each enquiry matches an application you made."},
    {"id": "unfamiliar-account", "title": "If you do not recognise an account", "topics": ["report", "disputes", "safety"],
     "body": "If your report lists an account you do not recognise, check the name and dates against your own records first. "
             "If it still looks wrong, raise a dispute with the bureau and keep a copy of what you send."},
    {"id": "debt-burden-ratio", "title": "What is a debt-burden ratio?", "topics": ["income", "budget"],
     "body": "A debt-burden ratio compares the money you spend on debt payments with the money you bring in. A lower ratio "
             "leaves more room in your monthly budget. Looking at your recurring payments is a good first step."},
    {"id": "small-cash-buffer", "title": "Building a small cash buffer", "topics": ["budget", "recommendations"],
     "body": "Setting aside a small amount each month helps you cope with an unexpected bill without missing a loan payment. "
             "Start with an amount that fits your budget and keep it apart from everyday spending."},
    {"id": "reading-your-report", "title": "Reading your credit report", "topics": ["report"],
     "body": "Your report lists your accounts, balances, payment history and enquiries, with the date each bureau produced it. "
             "Each bureau shows its own record, so details can differ between bureaus. Compare each entry with your own records."},
    {"id": "keep-payment-records", "title": "Keep proof of your payments", "topics": ["payments", "disputes"],
     "body": "Keep receipts or transfer alerts for your loan payments. If something on your report looks wrong, proof of payment "
             "makes it much easier to put right."},
    {"id": "protect-your-details", "title": "Protect your identity details", "topics": ["safety"],
     "body": "Your BVN, OTP and card PIN are for you only. CreditCliq and your lenders will not ask you to send them in chat or "
             "by message. If someone asks, do not share them."},
]


def render_item(d: dict[str, Any], owner: str = "SAMPLE OWNER", reviewer: str = "SAMPLE REVIEWER",
                reviewed_on: str = "2026-11-01", status: str = "approved", sample: bool = True) -> str:
    lines = ["---", f"id: {d['id']}", f"title: {d['title']}", f"topics: [{', '.join(d['topics'])}]", f"owner: {owner}",
             f"reviewer: {reviewer}", f"reviewed_on: {reviewed_on}", f"status: {status}", "language: en"]
    if sample:
        lines.append("sample: true")
    lines += ["sources:", f"  - {_SAMPLE_SRC}", "---", d["body"], ""]
    return "\n".join(lines)


def write_samples(folder: str) -> list[str]:
    os.makedirs(folder, exist_ok=True)
    paths = []
    for d in SAMPLE_ITEMS:
        p = os.path.join(folder, d["id"] + ".md")
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(render_item(d))
        paths.append(p)
    return paths


def check_model_dir(model_dir: str) -> list[str]:
    """Plain-language report on what is and is not in a model folder."""
    msgs: list[str] = []
    tok = os.path.join(model_dir, "tokenizer.json")
    msgs.append(("OK     " if os.path.exists(tok) else "MISSING") + f"  tokenizer.json  ({tok})")
    found = [os.path.join(d, n) for d in (os.path.join(model_dir, "onnx"), model_dir) if os.path.isdir(d)
             for n in sorted(os.listdir(d)) if n.endswith(".onnx")]
    if found:
        msgs += [f"OK       model file      ({p}, {os.path.getsize(p) / 1e6:.0f} MB)" for p in found]
    else:
        msgs.append(f"MISSING  an .onnx model file (looked in {model_dir} and {os.path.join(model_dir, 'onnx')})")
    try:
        import onnxruntime, tokenizers, numpy  # noqa: F401
        msgs.append("OK       packages        onnxruntime, tokenizers, numpy are installed")
    except ImportError as exc:
        msgs.append(f"MISSING  a package ({exc.name}):  pip install onnxruntime tokenizers numpy")
    return msgs


# =============================================================================
# 8. SELF-TEST
# =============================================================================

class _CountingEmbedder(HashEmbedder):
    def __init__(self):
        self.passages = 0
        self.queries = 0

    def embed_passages(self, texts):
        self.passages += len(texts)
        return super().embed_passages(texts)

    def embed_queries(self, texts):
        self.queries += len(texts)
        return super().embed_queries(texts)


def _sample_parsed(**kw) -> list[tuple[Item, list[Problem]]]:
    return [parse_item(render_item(d, **kw)) for d in SAMPLE_ITEMS]


def run_selftest() -> list[str]:
    fails: list[str] = []

    def expect(c: bool, what: str):
        if not c:
            fails.append(what)

    TODAY = date(2026, 11, 13)

    def codes(item: Item, **kw) -> set[str]:
        return {p.code for p in review_item(item, TODAY, **kw)}

    def good(**over) -> Item:
        d = dict(id="pay-on-time", title="Paying on time", body="Pay your loans on or before the due date. Contact your lender early if it is hard.",
                 topics=["payments"], owner="Ada", reviewer="Bayo", reviewed_on=date(2026, 10, 1), status="approved",
                 sources=["Consumer guidance note"], language="en", sample=False)
        d.update(over)
        return Item(**d)

    # ---- the parser ----
    text = render_item(SAMPLE_ITEMS[0])
    it, pr = parse_item(text)
    expect(not pr and it.id == "pay-on-time" and it.topics == ["payments", "recommendations"] and it.owner == "SAMPLE OWNER"
           and it.reviewed_on == date(2026, 11, 1) and it.sample and it.sources == [_SAMPLE_SRC] and it.body.startswith("Paying your loans"),
           f"parse a valid file {pr}")
    it2, pr2 = parse_item(text.replace("\n", "\r\n"))
    expect(not pr2 and it2.id == it.id and it2.body == it.body, "Windows line endings parse the same")
    expect(parse_item("no front matter")[1][0].code == "NO_FRONT_MATTER", "no front matter")
    expect(parse_item("---\nid: x\nno closing")[1][0].code == "NO_FRONT_MATTER", "unclosed front matter")
    expect(any(p.code == "UNKNOWN_FIELD" for p in parse_item(text.replace("owner:", "ownr:"))[1]), "a misspelt field is reported")
    expect(any(p.code == "BAD_DATE" for p in parse_item(text.replace("2026-11-01", "1st Nov"))[1]), "a bad date is reported")
    expect(parse_item(text.replace("[payments, recommendations]", "[\"a\", 'b']"))[0].topics == ["a", "b"], "quoted list items")

    # ---- the review gate ----
    expect(codes(good()) == set(), f"a good item passes: {codes(good())}")
    for label, item, want in [
        ("no owner", good(owner=""), "NO_OWNER"), ("no reviewer", good(reviewer=""), "NO_REVIEWER"),
        ("reviewer is owner", good(reviewer="ada"), "REVIEWER_IS_OWNER"), ("no date", good(reviewed_on=None), "NO_REVIEW_DATE"),
        ("future date", good(reviewed_on=date(2027, 1, 1)), "REVIEW_DATE_IN_FUTURE"), ("stale", good(reviewed_on=date(2025, 1, 1)), "STALE_REVIEW"),
        ("no source", good(sources=[]), "NO_SOURCE"), ("bad status", good(status="live"), "BAD_STATUS"), ("bad id", good(id="Pay On Time"), "BAD_ID"),
        ("no topics", good(topics=[]), "NO_TOPICS"), ("sample", good(sample=True), "SAMPLE_CONTENT"),
        ("too long", good(body="Pay on time. " * 80), "BODY_TOO_LONG"),
        ("scoring method", good(body="The algorithm weighs your payment history."), "BUREAU_METHOD"),
        ("how scores are calculated", good(body="Your score is calculated from your accounts."), "BUREAU_METHOD"),
        ("lender named", good(body="Ask GTBank about it."), "LENDER"), ("offer", good(body="Apply now for a special offer."), "OFFER"),
        ("promise", good(body="This is guaranteed to fix your report."), "PROMISE"), ("promise score", good(body="Paying will improve your score."), "PROMISE"),
        ("fraud verdict", good(body="This is fraud."), "FRAUD_LABEL"), ("pii", good(body="Send your BVN 12345678901 to us."), "PII"),
        ("long sentences", good(body=("This is a very long sentence that keeps going on and on without any sensible stopping point so that a "
                                       "reader would struggle to follow what it is trying to say to them at all today.")), "NOT_PLAIN"),
    ]:
        expect(want in codes(item), f"gate: {label} must raise {want}, got {codes(item)}")
    expect("SAMPLE_CONTENT" not in codes(good(sample=True), allow_samples=True), "samples pass when explicitly allowed")
    expect("STALE_REVIEW" not in codes(good(status="draft", reviewed_on=None)), "a draft is not held to review rules")

    # ---- load: what becomes searchable ----
    emb = _CountingEmbedder()
    kb = KnowledgeBase(":memory:", emb, today=lambda: TODAY)
    rep = kb.load(_sample_parsed())
    expect(kb.count() == 0 and len(rep.rejected) == len(SAMPLE_ITEMS) and all(any(p.code == "SAMPLE_CONTENT" for p in v) for v in rep.rejected.values()),
           "samples are NOT loaded by default")
    rep = kb.load(_sample_parsed(), allow_samples=True)
    expect(kb.count() == len(SAMPLE_ITEMS) and len(rep.loaded) == len(SAMPLE_ITEMS) and not rep.rejected, f"samples load when allowed: {rep.rejected}")
    n = emb.passages
    rep = kb.load(_sample_parsed(), allow_samples=True)
    expect(emb.passages == n and len(rep.unchanged) == len(SAMPLE_ITEMS) and not rep.loaded, "an unchanged item is not embedded again")
    changed = _sample_parsed()
    changed[0][0].body += " Set a reminder for the due date."
    rep = kb.load(changed, allow_samples=True)
    expect(rep.loaded == ["pay-on-time"] and emb.passages == n + 1, "only a changed item is embedded again")
    parsed = _sample_parsed()
    parsed[1] = parse_item(render_item(SAMPLE_ITEMS[1], status="retired"))
    rep = kb.load(parsed, allow_samples=True)
    expect("what-is-an-enquiry" in rep.removed and "what-is-an-enquiry" not in kb.ids() and "what-is-an-enquiry" not in rep.rejected, "a retired item leaves search on the next load")
    parsed = _sample_parsed()
    parsed[2] = parse_item(render_item(SAMPLE_ITEMS[2], reviewer="SAMPLE OWNER"))
    rep = kb.load(parsed, allow_samples=True)
    expect("unfamiliar-account" in rep.removed and "unfamiliar-account" in rep.rejected, "an item that stops passing the gate is removed")
    expect(kb.load(_sample_parsed(status="draft"), allow_samples=True).removed and kb.count() == 0, "drafts are never searchable")
    dup = _sample_parsed() + [parse_item(render_item(SAMPLE_ITEMS[0]))]
    rep = kb.load(dup, allow_samples=True)
    expect(any(p.code == "DUPLICATE_ID" for v in rep.rejected.values() for p in v), "a duplicate id is reported")
    kb.load(_sample_parsed(), allow_samples=True)

    # ---- search ----
    hits = kb.search("What is an enquiry on my report?")
    expect(hits and hits[0].id == "what-is-an-enquiry" and len(hits) <= TOP_K and all(h.score >= kb.threshold for h in hits), f"search finds the right item: {[h.id for h in hits]}")
    expect(hits == sorted(hits, key=lambda h: -h.score), "best first")
    expect(len(kb.search("payments report budget enquiry account", threshold=-1.0)) == TOP_K, "never more than 3")
    expect(kb.search("What is an enquiry on my report?", k=10) == kb.search("What is an enquiry on my report?", k=3), "k is capped at 3")
    expect(kb.search("tell me a joke about football") == [], "an unrelated question gets nothing")
    expect(all("safety" in h.topics for h in kb.search("someone asks for my OTP", topics=["safety"])), "topic filter")
    expect(kb.search("what is an enquiry", topics=["payments"]) == [] or all("payments" in h.topics for h in kb.search("what is an enquiry", topics=["payments"])), "topic filter excludes others")
    expect(kb.search("what is an enquiry", language="fr") == [], "language filter")
    q0 = emb.queries
    expect(kb.search("my bvn is 12345678901 what is an enquiry") == [] and kb.search("   ") == [] and emb.queries == q0, "an identifier or empty query is never embedded")
    expect(hits[0].sources == [_SAMPLE_SRC], "hits carry their sources")
    late = KnowledgeBase(":memory:", HashEmbedder(), today=lambda: TODAY)
    late.load(_sample_parsed(), allow_samples=True)
    late._today = lambda: TODAY + timedelta(days=400)   # a year later, nothing reloaded
    expect(late.search("what is an enquiry on my report", threshold=-1.0) == [], "an item whose review goes stale drops out of search by itself")
    other = KnowledgeBase(":memory:", HashEmbedder(), today=lambda: TODAY)
    other.load(_sample_parsed(), allow_samples=True)
    other._emb = type("E", (HashEmbedder,), {"id": "another-model"})()
    try:
        other.search("what is an enquiry")
        expect(False, "vectors from another embedder must not be compared")
    except RuntimeError:
        pass
    expect(kb.load(_sample_parsed(), allow_samples=True) is not None, "reload")
    path = os.path.join(tempfile.mkdtemp(), "kb.sqlite")
    k1 = KnowledgeBase(path, HashEmbedder(), today=lambda: TODAY)
    k1.load(_sample_parsed(), allow_samples=True)
    k1.close()
    expect(KnowledgeBase(path, HashEmbedder(), today=lambda: TODAY).search("what is an enquiry on my report")[0].id == "what-is-an-enquiry", "the file persists between runs")
    swap = KnowledgeBase(path, _CountingEmbedder(), today=lambda: TODAY)   # same embedder class but check re-embed on id change
    expect(swap.load(_sample_parsed(), allow_samples=True).unchanged, "same embedder id keeps vectors")

    # ---- evals and calibration ----
    ev = run_eval(kb)
    expect(ev["recall"] >= 0.85 and ev["false_accept_rate"] == 0.0, f"eval with the stand-in embedder: {ev['recall']}/{ev['false_accept_rate']} {[r['question'] for r in ev['rows'] if not r['ok']]}")
    cal = calibrate(kb)
    expect(cal["separable"] and cal["suggested_threshold"] is not None, f"calibration finds a gap: {cal}")

    # ---- the tool the Concierge calls ----
    tool = KnowledgeSearchTool(kb)
    out = tool({"query": "What is an enquiry on my report?"})
    expect(out["items"] and set(out["items"][0]) == {"id", "title", "body"} and len(out["items"]) <= 3, "tool shape")
    question = "What should I do to avoid missing loan payments?"
    out = tool({"query": question})   # exactly what the Concierge will search for
    expect(bool(out["items"]), "the Concierge's question finds guidance")
    expect(tool({}) == {"items": []} and tool({"query": 5}) == {"items": []} and tool({"query": "bvn 12345678901"}) == {"items": []}, "tool: bad input gives no items")

    # ---- the Concierge uses it, and can cite the item id ----
    import concierge as C
    from recommender import ScriptedLLM
    ans = json.dumps({"answer": "Your CRC score is 618 as of 31 Oct 2026. " + out["items"][0]["body"],
                      "citations": [{"bureau": "CRC", "date": "2026-10-31"}], "suggestions": [], "handoff": False,
                      "guidance": [out["items"][0]["id"]]})
    llm = ScriptedLLM([ans])
    r = C.Concierge(llm, fixtures=C.DEFAULT_FIXTURES, handlers={"knowledge.search": tool}).answer(
        {"question": question, "plan": "standard"})
    expect(r.origin == "model" and r.guidance == [out["items"][0]["id"]] and "knowledge.search" in r.tools_called,
           f"the Concierge answers from the knowledge base: {r.origin} {r.failed_attempt_codes}")
    expect(out["items"][0]["body"][:30] in llm.calls[0]["user"], "the model was shown the approved item")
    bad = json.loads(ans)
    bad["guidance"] = ["made-up-item"]
    r = C.Concierge(ScriptedLLM([json.dumps(bad), json.dumps(bad)]), fixtures=C.DEFAULT_FIXTURES, handlers={"knowledge.search": tool}).answer(
        {"question": question, "plan": "standard"})
    expect(r.origin == "template" and r.failed_attempt_codes and "GUIDANCE_NOT_RETURNED" in r.failed_attempt_codes[0], "a guidance id the search never returned is rejected")

    # ---- the e5 embedder, with stand-ins for the model and tokenizer ----
    try:
        import numpy as np

        class Enc:
            def __init__(self, ids, mask): self.ids, self.attention_mask = ids, mask

        class FakeTok:
            def __init__(self): self.seen, self.trunc, self.pad = [], None, None
            def token_to_id(self, t): return 1 if t == "<pad>" else None
            def enable_truncation(self, max_length): self.trunc = max_length
            def enable_padding(self, pad_id, pad_token): self.pad = (pad_id, pad_token)
            def encode_batch(self, texts):
                self.seen.extend(texts)
                width = max(len(t.split()) for t in texts)
                return [Enc([5 + i for i in range(len(t.split()))] + [1] * (width - len(t.split())), [1] * len(t.split()) + [0] * (width - len(t.split()))) for t in texts]

        class FakeSession:
            def __init__(self, names, pooled=False): self._n, self._p = names, pooled
            def get_inputs(self): return [type("I", (), {"name": n}) for n in self._n]
            def run(self, _, feeds):
                ids = feeds["input_ids"]
                self.feeds = feeds
                h = np.zeros((ids.shape[0], ids.shape[1], 4), dtype=np.float32)
                for b in range(ids.shape[0]):
                    for t in range(ids.shape[1]):
                        h[b, t] = [1.0, 0.0, 0.0, 0.0] if feeds["attention_mask"][b, t] else [0.0, 99.0, 99.0, 99.0]   # padding is garbage
                return [h[:, 0, :] * 3.0 if self._p else h]

        tok, sess = FakeTok(), FakeSession(["input_ids", "attention_mask"])
        e5 = E5Embedder.from_parts(sess, tok)
        v = e5.embed_queries(["what is an enquiry", "enquiry"])
        expect(tok.seen == ["query: what is an enquiry", "query: enquiry"], "e5: questions get the 'query: ' prefix")
        expect(tok.trunc == 512 and tok.pad == (1, "<pad>"), "e5: truncation and padding configured")
        expect(all(abs(sum(x * x for x in row) - 1.0) < 1e-6 for row in v) and v[0] == v[1] == [1.0, 0.0, 0.0, 0.0], "e5: padding ignored in the average, vectors have unit length")
        expect("token_type_ids" not in sess.feeds, "e5: only the inputs the model has are sent")
        e5.embed_passages(["a b"])
        expect(tok.seen[-1] == "passage: a b", "e5: documents get the 'passage: ' prefix")
        sess2 = FakeSession(["input_ids", "attention_mask", "token_type_ids"])
        E5Embedder.from_parts(sess2, FakeTok()).embed_queries(["x"])
        expect("token_type_ids" in sess2.feeds and not sess2.feeds["token_type_ids"].any(), "e5: token_type_ids sent as zeros when the model wants them")
        pooled = E5Embedder.from_parts(FakeSession(["input_ids", "attention_mask"], pooled=True), FakeTok()).embed_queries(["x y"])
        expect(abs(sum(x * x for x in pooled[0]) - 1.0) < 1e-6, "e5: an already-pooled export is accepted and normalised")
        many = E5Embedder.from_parts(FakeSession(["input_ids", "attention_mask"]), FakeTok()).embed_queries([f"w{i}" for i in range(40)])
        expect(len(many) == 40, "e5: more texts than one batch")
    except ImportError:
        pass

    # ---- model folder check, and the failure messages ----
    d = tempfile.mkdtemp()
    expect(any("MISSING" in m and "tokenizer" in m for m in check_model_dir(d)) and any("MISSING" in m and ".onnx" in m for m in check_model_dir(d)), "check-model reports missing files")
    os.makedirs(os.path.join(d, "onnx"))
    open(os.path.join(d, "tokenizer.json"), "w").write("{}")
    open(os.path.join(d, "onnx", "model.onnx"), "wb").write(b"x")
    expect(any(m.startswith("OK") and "model file" in m for m in check_model_dir(d)), "check-model finds the files")
    try:
        make_embedder("e5")
        expect(False, "e5 without a model dir must explain")
    except RuntimeError as exc:
        expect("--model-dir" in str(exc), "e5 without a model directory: a clear message")
    try:
        E5Embedder(tempfile.mkdtemp())
        expect(False, "an empty model dir must explain")
    except RuntimeError as exc:
        expect("check-model" in str(exc) or "pip install" in str(exc), "an empty model folder: a clear message")

    # ---- sample files round-trip through the folder reader ----
    folder = tempfile.mkdtemp()
    expect(len(write_samples(folder)) == len(SAMPLE_ITEMS) and len(read_folder(folder)) == len(SAMPLE_ITEMS), "samples are written as files and read back")
    rd = read_folder(folder)
    expect(all(not pr for _, pr in rd) and not any(review_item(i, TODAY, allow_samples=True) for i, _ in rd), "the written sample files pass the gate (when samples are allowed)")
    return fails


def _demo() -> None:
    kb = KnowledgeBase(":memory:", HashEmbedder(), today=lambda: date(2026, 11, 13))
    rep = kb.load(_sample_parsed(), allow_samples=True)
    print(f"loaded {len(rep.loaded)} sample items (stand-in embedder)\n")
    for q in ("What is an enquiry on my report?", "How can I avoid missing loan payments?", "Someone is asking me for my OTP",
              "Tell me a joke about football"):
        hits = kb.search(q)
        print(f"Q: {q}")
        print("   " + (", ".join(f"{h.id} ({h.score:.2f})" for h in hits) if hits else "nothing above the threshold, so the Concierge gets no guidance"))
    ev = run_eval(kb)
    print(f"\neval: right item found for {ev['recall']:.0%} of questions; unrelated questions wrongly answered: {ev['false_accept_rate']:.0%}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="kb.py", description="CreditCliq Knowledge Base")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--write-samples", metavar="DIR", help="write the sample items as Markdown files")
    mode.add_argument("--check", metavar="DIR", help="run the review gate on a folder")
    mode.add_argument("--load", metavar="DIR", help="embed and store a folder of items")
    mode.add_argument("--search", metavar="TEXT")
    mode.add_argument("--eval", action="store_true")
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--check-model", metavar="DIR")
    mode.add_argument("--download-model", metavar="DIR")
    mode.add_argument("--print-schema", action="store_true", help="the PostgreSQL + pgvector schema for production")
    ap.add_argument("--db", default="knowledge.sqlite")
    ap.add_argument("--embedder", choices=["hash", "e5"], default="hash")
    ap.add_argument("--model-dir")
    ap.add_argument("--include-samples", action="store_true")
    ap.add_argument("--threshold", type=float)
    args = ap.parse_args(argv)

    if args.print_schema:
        print(SCHEMA_PG)
        return 0
    if args.demo:
        _demo()
        return 0
    if args.write_samples:
        for p in write_samples(args.write_samples):
            print("wrote", p)
        print("\nThese are SAMPLES (sample: true). Real items need a real owner, a different reviewer, a review date and sources.")
        return 0
    if args.check_model:
        msgs = check_model_dir(args.check_model)
        print("\n".join(msgs))
        return 1 if any(m.startswith("MISSING") for m in msgs) else 0
    if args.download_model:
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Xenova/multilingual-e5-small", local_dir=args.download_model,
                              allow_patterns=["tokenizer.json", "onnx/model.onnx", "onnx/model_quantized.onnx", "config.json"])
        except Exception as exc:
            print(f"Download failed ({type(exc).__name__}). Run  pip install huggingface_hub  and try again, or open "
                  "https://huggingface.co/Xenova/multilingual-e5-small , download tokenizer.json and onnx/model.onnx "
                  f"into {args.download_model}\\onnx\\ and {args.download_model}\\.")
            return 1
        print("\n".join(check_model_dir(args.download_model)))
        return 0
    if args.check:
        bad = 0
        for item, pr in read_folder(args.check):
            problems = pr + review_item(item, date.today(), args.include_samples)
            print(f"{'OK    ' if not problems else 'FAILED'}  {item.id or os.path.basename(item.path)}")
            for p in problems:
                print(f"          - {p}")
            bad += bool(problems)
        return 1 if bad else 0
    if not (args.load or args.search or args.eval or args.calibrate):
        fails = run_selftest()
        if fails:
            print("FAILED:", *fails, sep="\n - ")
            return 1
        print("self-test: file format, review gate, load/retire/re-embed, search and threshold, stale items, identifiers never embedded, "
              "evals and calibration, the Concierge tool with citable ids, and the e5 embedder logic (with stand-ins) verified")
        print("ALL PASSED")
        return 0

    try:
        emb = make_embedder(args.embedder, args.model_dir)
    except RuntimeError as exc:
        print(exc)
        return 1
    kb = KnowledgeBase(args.db, emb, args.threshold)
    if args.load:
        rep = kb.load(read_folder(args.load), args.include_samples)
        print(f"embedder {emb.id}: embedded {len(rep.loaded)}, unchanged {len(rep.unchanged)}, removed {len(rep.removed)}, "
              f"not searchable {len(rep.rejected)}")
        for k, v in rep.rejected.items():
            print(f"  not searchable: {k}")
            for p in v:
                print(f"      - {p}")
        return 0
    if args.search:
        hits = kb.search(args.search)
        for h in hits:
            print(f"{h.score:.3f}  {h.id}: {h.title}\n        {h.body}")
        if not hits:
            print(f"nothing above the threshold ({kb.threshold})")
        return 0
    if args.calibrate:
        print(json.dumps(calibrate(kb), indent=2))
        return 0
    ev = run_eval(kb)
    for r in ev["rows"]:
        print(f"  {'ok  ' if r['ok'] else 'FAIL'} {r['question'][:52]:<54} wanted {r['want'] or 'nothing'}  got {[(i, round(s, 2)) for i, s in r['got']] or 'nothing'}")
    print(f"\nthreshold {kb.threshold}: right item found {ev['recall']:.0%} ({ev['positives']} questions); unrelated questions answered {ev['false_accept_rate']:.0%} ({ev['negatives']} questions)")
    return 0 if ev["recall"] >= 0.8 and ev["false_accept_rate"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))