-- The home for receipts that can never be processed.
CREATE TABLE IF NOT EXISTS bad_receipts (
    id          BIGSERIAL PRIMARY KEY,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_id    TEXT,
    raw_receipt JSONB NOT NULL,
    error       TEXT NOT NULL
);

-- Makes "show me the last 24 hours" and old-row clean-up quick.
CREATE INDEX IF NOT EXISTS idx_bad_receipts_received_at
    ON bad_receipts (received_at);