"""
enforce.py

The manager. She reads the totals, compares them with the limits,
and decides if the system should slow down or stop paid things.
She can only LOWER spending. She never raises a budget
or changes a price.
"""

from datetime import datetime, timezone
from decimal import Decimal

import yaml
from redis.exceptions import RedisError

from .counters import get_day_spend, get_quota_used


def load_budgets(path):
    # Read the whole budget file into a dictionary.
    with open(path) as file:
        return yaml.safe_load(file)


def decide_company_signal(spend_today, company):
    # Compare today's spend with the company's daily budget.
    budget = Decimal(str(company["daily_budget_ngn"]))
    if spend_today >= budget * Decimal(str(company["pause_spend_at"])):
        return "pause_spend"   # stop the paid things
    if spend_today >= budget * Decimal(str(company["tier_cap_at"])):
        return "tier_cap"      # use a cheaper brain
    return None                # all fine


def quota_reached(used, plan, budgets):
    # Has this person used up their plan's monthly allowance?
    limits = budgets["plans"].get(plan)
    if limits is None:
        # Plan not in our file. When unsure, spend less.
        return True, "unknown_plan"
    allowed = Decimal(str(limits["monthly_bureau_pulls"]))
    if used >= allowed:
        return True, "monthly_allowance_used"
    return False, None


def check_limits(client, budgets, subscriber_ref, plan, now=None):
    # The manager's full check for one person, right now.
    now = now or datetime.now(timezone.utc)

    try:
        spend_today = get_day_spend(client, now)
        pulls_used = get_quota_used(client, subscriber_ref, now, "bureau_pulls")
    except RedisError:
        # Cannot read the whiteboard. When unsure, spend less.
        return {"signal": "pause_spend", "quota_reached": True,
                "reason": "counters_unreadable"}

    signal = decide_company_signal(spend_today, budgets["company"])
    reached, reason = quota_reached(pulls_used, plan, budgets)
    return {"signal": signal, "quota_reached": reached, "reason": reason}


def send_signal(result):
    # PLACEHOLDER. We do not yet know how the Model gateway listens,
    # so for now we only print what we WOULD send.
    if result["signal"]:
        print("Would tell the gateway:", result["signal"])
        print("Would alert the Business Lead")


# Practice run. It needs no Redis.
if __name__ == "__main__":
    budgets = load_budgets("config/budgets.yaml")
    company = budgets["company"]

    # Test the company budget at three spend levels.
    for spend in ["7000", "8500", "10000"]:
        print(f"Spend {spend} ->", decide_company_signal(Decimal(spend), company))

    # Test the person's allowance.
    print("Rebound, 10 used ->", quota_reached(Decimal("10"), "rebound", budgets))
    print("Rebound, 3 used  ->", quota_reached(Decimal("3"), "rebound", budgets))
    print("Unknown plan     ->", quota_reached(Decimal("0"), "mystery", budgets))

    # Test what happens when Redis cannot be read.
    class BrokenRedis:
        def get(self, key):
            raise RedisError("cannot reach redis")

    print("Redis down       ->", check_limits(BrokenRedis(), budgets, "sub_test_001", "rebound"))