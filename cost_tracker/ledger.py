"""
ledger.py

The bookkeeper. She writes one line in the big book (PostgreSQL)
for every cost. The ledger is the official record.
"""

from .pricing import load_prices, price_event, to_decimal
from .usage_event import UsageEvent

# The columns of the cost_ledger table, in a fixed order.
# These names must match sql/create_ledger.sql exactly.
COLUMNS = [
    "event_id", "occurred_at", "source", "tier", "item", "quantity", "unit",
    "plan", "subscriber_ref", "agent",
    "original_cost", "original_currency", "fx_rate", "cost_ngn",
]

# The instruction we send to PostgreSQL.
# Each %s is an empty slot. The values are filled in safely later.
# ON CONFLICT DO NOTHING means: if this receipt id is already in the
# book, skip it. This stops the same cost being counted twice.
INSERT_SQL = f"""
    INSERT INTO cost_ledger ({", ".join(COLUMNS)})
    VALUES ({", ".join(["%s"] * len(COLUMNS))})
    ON CONFLICT (event_id) DO NOTHING
"""


def build_ledger_row(event: UsageEvent, priced: dict) -> dict:
    # Put the receipt and the price clerk's result into one line.
    return {
        # From the receipt
        "event_id": event.event_id,
        "occurred_at": event.occurred_at,
        "source": event.source,
        "tier": event.tier,
        "item": event.item,
        "quantity": to_decimal(event.quantity),
        "unit": event.unit,
        "plan": event.plan,
        "subscriber_ref": event.subscriber_ref,
        "agent": event.agent,
        # From the price clerk
        "original_cost": priced["original_cost"],
        "original_currency": priced["original_currency"],
        "fx_rate": priced["fx_rate"],
        "cost_ngn": priced["cost_ngn"],
    }


def write_ledger_row(connection, row: dict) -> bool:
    # Write one line into the big book.
    # Returns True if a new line was written,
    # False if this receipt was already there.
    values = [row[column] for column in COLUMNS]
    with connection.cursor() as cursor:
        cursor.execute(INSERT_SQL, values)
        written = cursor.rowcount == 1
    connection.commit()
    return written


# Dry run. No database needed. It only builds the line and prints it.
if __name__ == "__main__":
    prices = load_prices("config/prices.yaml")

    event = UsageEvent(
        source="bureau_gateway", item="crc_report",
        quantity=1, unit="bureau_pulls", plan="standard",
        subscriber_ref="sub_8f3a21", agent="concierge",
    )
    priced = price_event(event, prices, usd_to_ngn=1347)
    row = build_ledger_row(event, priced)

    print("LEDGER LINE:")
    for name, value in row.items():
        print(f"  {name}: {value}")

    print("\nSQL THAT WOULD RUN:")
    print(INSERT_SQL)