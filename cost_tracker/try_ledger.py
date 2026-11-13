"""
try_ledger.py

A practice run. It writes one test receipt into the real ledger,
then tries to write the same receipt again.
"""

from .db import get_connection
from .ledger import build_ledger_row, write_ledger_row
from .pricing import load_prices, price_event
from .usage_event import UsageEvent

if __name__ == "__main__":
    prices = load_prices("config/prices.yaml")

    # One test receipt: a single CRC report pull.
    # "sub_test_001" marks it clearly as practice data.
    event = UsageEvent(
        source="bureau_gateway", item="crc_report",
        quantity=1, unit="bureau_pulls", plan="standard",
        subscriber_ref="sub_test_001", agent="practice",
    )

    # Price it, then build the ledger line.
    priced = price_event(event, prices, usd_to_ngn=1347)
    row = build_ledger_row(event, priced)

    with get_connection() as connection:
        # Write the line for the first time.
        first = write_ledger_row(connection, row)
        print("First write, new line written?", first)

        # Try to write the SAME receipt again.
        second = write_ledger_row(connection, row)
        print("Second write, new line written?", second)