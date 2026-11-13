"""
pricing.py

The price clerk. She takes one receipt (a UsageEvent)
and works out what it cost us, in naira.
All money maths uses Decimal, so there are no tiny rounding errors.
"""

from decimal import Decimal, ROUND_HALF_UP

import yaml  # reads the price file

# The dot means: from the same folder as this file.
from .usage_event import UsageEvent


class UnknownItemError(Exception):
    """Raised when the price list has no price for an item."""


def load_prices(path):
    # Open the price file and return the "items" part as a dictionary.
    with open(path) as file:
        return yaml.safe_load(file)["items"]


def to_decimal(number):
    # Turn a number into an exact Decimal.
    # We go through text (str) first so 0.5 stays exactly 0.5.
    return Decimal(str(number))


def price_event(event: UsageEvent, prices: dict, usd_to_ngn) -> dict:
    # Step 1: find this item in the price list.
    entry = prices.get(event.item)
    if entry is None:
        # Stop with an error. A silent zero would hide real spending.
        raise UnknownItemError(f"No price for item: {event.item}")

    # Step 2: check the receipt counts in the same unit as the price.
    if entry["unit"] != event.unit:
        raise ValueError(
            f"Unit mismatch for {event.item}: "
            f"receipt says {event.unit}, price list says {entry['unit']}"
        )

    # Step 3: work out the cost in the ORIGINAL currency, using exact numbers.
    quantity = to_decimal(event.quantity)
    per = to_decimal(entry["per"])
    price = to_decimal(entry["price"])
    original_cost = (quantity / per) * price

    # Step 4: pick the exchange rate. Naira needs no conversion.
    if entry["currency"] == "NGN":
        fx_rate = Decimal("1")
    elif entry["currency"] == "USD":
        fx_rate = to_decimal(usd_to_ngn)
    else:
        raise ValueError(f"Unknown currency: {entry['currency']}")

    # Step 5: convert to naira, round to 4 decimal places,
    # and keep everything needed to check the maths later.
    cost_ngn = (original_cost * fx_rate).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    )
    return {
        "event_id": event.event_id,
        "original_cost": original_cost,
        "original_currency": entry["currency"],
        "fx_rate": fx_rate,
        "cost_ngn": cost_ngn,
    }


# Quick test. Only runs when this file is run directly.
if __name__ == "__main__":
    prices = load_prices("config/prices.yaml")
    todays_rate = 1347  # practice rate, naira per dollar

    # Test 1: an external AI call that sent 2,000 tokens in.
    ai_event = UsageEvent(
        source="model_gateway", tier="external", item="external-llm:input",
        quantity=2000, unit="tokens", plan="standard",
        subscriber_ref="sub_8f3a21",
    )
    print(price_event(ai_event, prices, todays_rate))

    # Test 2: one CRC report pull.
    report_event = UsageEvent(
        source="bureau_gateway", item="crc_report",
        quantity=1, unit="bureau_pulls", plan="standard",
        subscriber_ref="sub_8f3a21",
    )
    print(price_event(report_event, prices, todays_rate))