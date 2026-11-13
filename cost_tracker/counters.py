"""
counters.py

The tally keeper. She keeps quick running totals on the whiteboard (Redis).
The ledger (PostgreSQL) is still the official record.
If Redis is ever lost, the totals can be rebuilt from the ledger.
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

from .ledger import build_ledger_row
from .pricing import load_prices, price_event
from .redis_client import get_redis
from .usage_event import UsageEvent

# Every key starts with this, so we never mix with other projects' keys.
PREFIX = "cost_tracker"

# Redis adds whole numbers exactly, so we store money and amounts as
# whole numbers: money in 1/10,000 of a naira, amounts in thousandths.
MONEY_SCALE = 10_000
QUANTITY_SCALE = 1_000


def to_whole(value, scale):
    # Turn an exact Decimal into a whole number, e.g. 300 naira -> 3000000.
    return int((Decimal(value) * scale).to_integral_value(rounding=ROUND_HALF_UP))


def from_whole(number, scale):
    # Turn the whole number back into a normal Decimal.
    return Decimal(number) / scale


def day_key(when):
    # The label for the company's spend on one day.
    return f"{PREFIX}:spend:day:{when:%Y-%m-%d}"


def quota_key(subscriber_ref, when, unit):
    # The label for one person's usage of one unit in one month.
    return f"{PREFIX}:quota:{subscriber_ref}:{when:%Y-%m}:{unit}"


def add_to_counters(client, row):
    # Add one ledger line to the running totals.
    when = row["occurred_at"]

    # A pipeline sends both additions together, so we never
    # update one total and forget the other.
    pipe = client.pipeline()
    pipe.incrby(day_key(when), to_whole(row["cost_ngn"], MONEY_SCALE))
    pipe.incrby(
        quota_key(row["subscriber_ref"], when, row["unit"]),
        to_whole(row["quantity"], QUANTITY_SCALE),
    )
    pipe.execute()


def get_day_spend(client, when):
    # How much naira has the company spent on this day?
    raw = client.get(day_key(when))
    return from_whole(int(raw), MONEY_SCALE) if raw else Decimal("0")


def get_quota_used(client, subscriber_ref, when, unit):
    # How much of this unit has this person used this month?
    raw = client.get(quota_key(subscriber_ref, when, unit))
    return from_whole(int(raw), QUANTITY_SCALE) if raw else Decimal("0")


# Practice run. It uses a fake date (year 2000) so it never touches
# a real day's total, and it deletes its own keys at the end.
if __name__ == "__main__":
    client = get_redis()
    prices = load_prices("config/prices.yaml")
    fake_time = datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)
    who = "sub_test_001"

    def make_row(item):
        event = UsageEvent(
            source="bureau_gateway", item=item, quantity=1,
            unit="bureau_pulls", plan="standard", subscriber_ref=who,
            occurred_at=fake_time,
        )
        priced = price_event(event, prices, usd_to_ngn=1347)
        return build_ledger_row(event, priced)

    # Receipt 1: a CRC report (300 naira)
    add_to_counters(client, make_row("crc_report"))
    print("After CRC report: spend =", get_day_spend(client, fake_time),
          "| pulls used =", get_quota_used(client, who, fake_time, "bureau_pulls"))

    # Receipt 2: a FICO score (50 naira)
    add_to_counters(client, make_row("fico_score"))
    print("After FICO score: spend =", get_day_spend(client, fake_time),
          "| pulls used =", get_quota_used(client, who, fake_time, "bureau_pulls"))

    # Clean up our own practice keys.
    client.delete(day_key(fake_time), quota_key(who, fake_time, "bureau_pulls"))
    print("Cleaned up:", client.get(day_key(fake_time)) is None)