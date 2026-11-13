"""
usage_event.py

A "usage event" is a small receipt.
Every time something costs money, the part that spent it
(the Model gateway, the local model server, or the bureau gateway)
sends one of these receipts to the Cost Tracker.

This file only describes WHAT a receipt looks like and checks it.
It does not work out any cost yet. That comes later in pricing.py.
"""

import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UsageEvent(BaseModel):
    # "forbid" means: if someone sends a field we did not expect,
    # reject the receipt instead of quietly accepting it.
    model_config = ConfigDict(extra="forbid")

    # A unique number for this receipt. Made for us if not given.
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # When it happened. We use UTC time so everyone agrees.
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Who is sending this receipt?
    source: Literal["model_gateway", "local_inference", "bureau_gateway"]

    # Which "brain" answered? Bureau pulls have no brain, so this can be empty.
    tier: Optional[Literal["cache", "template", "local", "external"]] = None

    # What was used. Examples: a model name, or a bureau name.
    # pricing.py will use this to look up the price later.
    item: str

    # How much was used, and in what unit.
    # We record the AMOUNT used, not the cost.
    # The cost is worked out later from the price list.
    quantity: float = Field(ge=0)  # ge=0 means "zero or more", never negative
    unit: Literal["tokens", "cpu_seconds", "bureau_pulls"]

    # The person's plan (for example "entry", "standard", "top").
    plan: str

    # An anonymous id for the person. NEVER a BVN, name or phone number.
    subscriber_ref: str

    # Which agent asked for this (for example "concierge"). Can be empty.
    agent: Optional[str] = None

    @field_validator("subscriber_ref")
    @classmethod
    def must_not_look_like_a_bvn(cls, value: str) -> str:
        # A BVN is 11 digits. If the id is all digits and 11 long,
        # it may be a real BVN, so we refuse it.
        if value.isdigit() and len(value) == 11:
            raise ValueError("subscriber_ref looks like a BVN. Use an anonymous id.")
        return value


# This part only runs when you run this file directly.
# It is a quick test so we can see it working.
if __name__ == "__main__":
    # A good receipt: the local model used 3.2 seconds of CPU.
    good = UsageEvent(
        source="local_inference",
        tier="local",
        item="granite-4.1-3b",
        quantity=3.2,
        unit="cpu_seconds",
        plan="standard",
        subscriber_ref="sub_8f3a21",
        agent="concierge",
    )
    print("GOOD RECEIPT:")
    print(good.model_dump_json(indent=2))

    # A bad receipt: the id looks like a BVN, so it should be refused.
    print("\nBAD RECEIPT (should be refused):")
    try:
        UsageEvent(
            source="model_gateway",
            tier="external",
            item="external-llm",
            quantity=2000,
            unit="tokens",
            plan="top",
            subscriber_ref="12345678901",
        )
    except ValueError as error:
        print("Refused, as expected:", error)