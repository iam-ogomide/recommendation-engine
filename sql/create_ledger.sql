-- create_ledger.sql
-- The big book: one row for every cost. This is the official record.

CREATE TABLE IF NOT EXISTS cost_ledger (
    -- From the receipt (UsageEvent)
    event_id          TEXT PRIMARY KEY,
    occurred_at       TIMESTAMPTZ NOT NULL,
    source            TEXT NOT NULL,
    tier              TEXT,
    item              TEXT NOT NULL,
    quantity          NUMERIC NOT NULL,
    unit              TEXT NOT NULL,
    plan              TEXT NOT NULL,
    subscriber_ref    TEXT NOT NULL,
    agent             TEXT,

    -- From the price clerk (pricing.py)
    original_cost     NUMERIC NOT NULL,
    original_currency TEXT NOT NULL,
    fx_rate           NUMERIC NOT NULL,
    cost_ngn          NUMERIC NOT NULL,

    -- When the bookkeeper wrote the line
    recorded_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Speeds up "add up this person's costs" and "add up today's costs".
CREATE INDEX IF NOT EXISTS idx_ledger_subscriber_time
    ON cost_ledger (subscriber_ref, occurred_at);

CREATE INDEX IF NOT EXISTS idx_ledger_time
    ON cost_ledger (occurred_at);