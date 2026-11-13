"""
consumer.py

The door clerk. For now it has two jobs:
- handle_receipt: takes one receipt through the whole line.
- handle_safely: does the same, but saves a bad receipt in the
  bad_receipts table so it cannot stop the line.

Collecting receipts from the event bus will be added later.
"""

from .bad_receipts import save_bad_receipt
from .counters import add_to_counters
from .enforce import check_limits, send_signal
from .ledger import build_ledger_row, write_ledger_row
from .pricing import price_event, UnknownItemError
from .usage_event import UsageEvent


def handle_receipt(raw, prices, budgets, connection, redis_client, usd_to_ngn):
    # Step 1: check the receipt is filled in properly.
    # A bad receipt stops here with an error.
    event = UsageEvent.model_validate(raw)

    # Step 2: price it in naira.
    priced = price_event(event, prices, usd_to_ngn)

    # Step 3: write it in the ledger (the official record).
    row = build_ledger_row(event, priced)
    is_new = write_ledger_row(connection, row)

    # If the ledger already had this receipt, stop.
    # Counting it again would double the totals.
    if not is_new:
        return {"status": "duplicate", "event_id": event.event_id}

    # Step 4: update the running totals in Redis.
    add_to_counters(redis_client, row)

    # Step 5: the manager checks the limits.
    limits = check_limits(redis_client, budgets, event.subscriber_ref, event.plan)

    # Step 6: tell the gateway if a limit was hit.
    send_signal(limits)

    return {
        "status": "recorded",
        "event_id": event.event_id,
        "cost_ngn": row["cost_ngn"],
        "limits": limits,
    }


def handle_safely(raw, prices, budgets, connection, redis_client, usd_to_ngn):
    # Same as handle_receipt, but one bad receipt cannot stop the line.
    try:
        return handle_receipt(
            raw, prices, budgets, connection, redis_client, usd_to_ngn
        )
    except (ValueError, UnknownItemError) as error:
        # The receipt itself is wrong. Save it in the bad_receipts table
        # and carry on with the next one.
        connection.rollback()  # clear any half-finished database work first
        save_bad_receipt(connection, raw, str(error))
        return {"status": "set_aside"}
    # Other errors, like the database being down, are NOT caught here,
    # so they are not hidden and the receipt can be tried again later.


# Practice run. It records one test receipt, tries the same receipt again,
# then removes its own test data.
if __name__ == "__main__":
    from datetime import datetime, timezone

    from .counters import day_key, quota_key
    from .db import get_connection
    from .enforce import load_budgets
    from .pricing import load_prices
    from .redis_client import get_redis

    prices = load_prices("config/prices.yaml")
    budgets = load_budgets("config/budgets.yaml")
    redis_client = get_redis()
    fake_time = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)

    # A test receipt as plain data. The year 2000 keeps it away
    # from any real day's totals.
    receipt = {
        "event_id": "test-consumer-0001",
        "occurred_at": "2000-01-01T12:00:00+00:00",
        "source": "bureau_gateway",
        "item": "crc_report",
        "quantity": 1,
        "unit": "bureau_pulls",
        "plan": "rebound",
        "subscriber_ref": "sub_test_001",
        "agent": "practice",
    }

    def cleanup(connection):
        # Remove only our own test data.
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM cost_ledger WHERE event_id = %s",
                (receipt["event_id"],),
            )
        connection.commit()
        redis_client.delete(
            day_key(fake_time),
            quota_key("sub_test_001", fake_time, "bureau_pulls"),
        )

    with get_connection() as connection:
        cleanup(connection)  # start clean
        print("First :", handle_receipt(
            receipt, prices, budgets, connection, redis_client, 1347))
        print("Second:", handle_receipt(
            receipt, prices, budgets, connection, redis_client, 1347))
        cleanup(connection)  # leave no test data behind
        print("Cleaned up")