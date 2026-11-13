# Cost Tracker

The Cost Tracker records what each AI call and bureau pull costs CreditChek,
keeps running totals, and checks them against limits.
Costs are kept in naira. Users are not charged by this code.

## How a receipt travels

A usage event (a "receipt") is sent by a gateway. `consumer.py` runs it through:

1. `usage_event.py` checks the receipt is filled in properly
2. `pricing.py` works out the cost in naira
3. `ledger.py` writes it in PostgreSQL (the official record)
4. `counters.py` updates the running totals in Redis
5. `enforce.py` checks the limits and decides on a signal
6. `send_signal` passes the signal on (right now it only prints)

If the ledger already has a receipt, it is skipped, so nothing is counted twice.
A receipt that can never be processed is saved in the `bad_receipts` table.

## Files

| File | Job |
|---|---|
| `usage_event.py` | The receipt form and its rules (rejects BVN-like ids) |
| `pricing.py` | Prices a receipt in naira using `config/prices.yaml` |
| `ledger.py` | Writes one ledger row per cost |
| `counters.py` | Redis totals: company spend per day, usage per person per month |
| `enforce.py` | Compares totals with `config/budgets.yaml`, decides `tier_cap` or `pause_spend` |
| `consumer.py` | Ties it together: `handle_receipt` and `handle_safely` |
| `bad_receipts.py` | Saves bad receipts (BVN-like ids are hidden first) |
| `db.py`, `redis_client.py` | Connections, using the login in `.env` |
| `try_ledger.py`, `try_bad_receipts.py` | Practice scripts, not part of the real flow |

Other folders: `config/` (prices and budgets) and `sql/` (table definitions).

## Set up

1. `pip install -r requirements.txt`
2. Create a `.env` file in the project root with these names and your own values. Never commit it.
   `DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, REDIS_HOST, REDIS_PORT, REDIS_USERNAME, REDIS_PASSWORD, REDIS_SSL`
   (`DB_SSLMODE` is optional)
3. Create the tables by running `sql/create_ledger.sql` and `sql/create_bad_receipts.sql`.

## Try it

Run from the project root:

```
python -m cost_tracker.usage_event
python -m cost_tracker.pricing
python -m cost_tracker.ledger
python -m cost_tracker.enforce
python -m cost_tracker.consumer
python -m cost_tracker.try_bad_receipts
```

`db`, `redis_client`, `counters` and `try_ledger` also run on their own.
`enforce`, `ledger` (dry run) and `pricing` need no database.

## Placeholders to replace

All numbers in `config/prices.yaml` and `config/budgets.yaml` are made up.
The plan names `free` and `rebound` are guesses. Prices must match the item names
on receipts (for example `crc_report`, `fico_score`).

## Test data left in the databases

On purpose, test rows were not deleted:
`sub_test_001`, event ids starting `test-`, and date 2000-01-01.
To see real costs only: `SELECT * FROM cost_ledger WHERE event_id NOT LIKE 'test-%';`

## Still to do

Waiting on answers from the team:
1. Event bus (NATS JetStream): access, mailbox name, a sample receipt, and who builds the collector loop that passes receipts to `handle_safely`
2. How the Model gateway receives `tier_cap` and `pause_spend` (a message or a Redis flag)
3. Exchange rate: who provides it, from what source, how often. It is passed in by hand for now
4. Real prices, plan names and limits, and who owns `budgets.yaml`. Also whether limits are daily or monthly, and in naira or pulls
5. Whether limits are checked before each answer or after each cost

Decisions and extras:
6. Redis: confirm no-eviction, and decide how long counters live
7. `handle_safely` sets aside every `ValueError`, so a config typo can fill the bad receipts. Watch for a sudden rise
8. Budget days use UTC