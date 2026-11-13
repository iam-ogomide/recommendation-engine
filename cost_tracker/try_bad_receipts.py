"""
try_bad_receipts.py

Practice: four receipts. Two are bad, and the bad ones are saved
in the bad_receipts table. Nothing is deleted afterwards.
Each run uses fresh ids, so runs never clash with each other.
"""

from datetime import datetime, timezone

from .consumer import handle_safely
from .db import get_connection
from .enforce import load_budgets
from .pricing import load_prices
from .redis_client import get_redis

if __name__ == "__main__":
    prices = load_prices("config/prices.yaml")
    budgets = load_budgets("config/budgets.yaml")
    redis_client = get_redis()

    # A stamp for this run, so every receipt id is new each time.
    run = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    def make(number, item, quantity, who="sub_test_001"):
        return {
            "event_id": f"test-batch-{run}-{number:04d}",
            "occurred_at": "2000-01-01T12:00:00+00:00",
            "source": "bureau_gateway",
            "item": item,
            "quantity": quantity,
            "unit": "bureau_pulls",
            "plan": "rebound",
            "subscriber_ref": who,
        }

    mailbox = [
        make(1, "crc_report", 1),
        make(2, "crc_report", -5),                    # BAD: negative quantity
        make(3, "fico_score", 1),
        make(4, "crc_report", 1, who="12345678901"),  # BAD: looks like a BVN
    ]
    ids = [receipt["event_id"] for receipt in mailbox]

    with get_connection() as connection:
        for receipt in mailbox:
            result = handle_safely(
                receipt, prices, budgets, connection, redis_client, 1347
            )
            print(result["status"], "-", receipt["event_id"])

        # Look at what was saved in the bad_receipts table for THIS run.
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT event_id, raw_receipt->>'subscriber_ref', error "
                "FROM bad_receipts WHERE event_id = ANY(%s) ORDER BY event_id",
                (ids,),
            )
            rows = cursor.fetchall()

        print("Saved in bad_receipts:", len(rows))
        for event_id, ref, error in rows:
            print("  ", event_id, "| id shown:", ref, "|", error.splitlines()[0])

        leaked = any("12345678901" in f"{ref} {error}" for _, ref, error in rows)
        print("BVN anywhere in saved data?", leaked)