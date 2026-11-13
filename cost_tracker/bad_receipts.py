"""
bad_receipts.py

The home of the bad pile. A receipt that can never be processed
is saved here with the reason, so a person can look at it later.
"""

import re

from psycopg.types.json import Jsonb


def hide_bvn_like_id(raw):
    # If the person's id looks like a BVN (11 digits), hide it before saving.
    if isinstance(raw, dict):
        ref = raw.get("subscriber_ref")
        if isinstance(ref, str) and ref.isdigit() and len(ref) == 11:
            raw = {**raw, "subscriber_ref": "[HIDDEN]"}
    return raw


def save_bad_receipt(connection, raw, error):
    # The error text can repeat the bad value, so hide 11-digit numbers there too.
    safe_error = re.sub(r"\b\d{11}\b", "[HIDDEN]", error)
    event_id = raw.get("event_id") if isinstance(raw, dict) else None

    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO bad_receipts (event_id, raw_receipt, error) "
            "VALUES (%s, %s, %s)",
            (event_id, Jsonb(hide_bvn_like_id(raw)), safe_error),
        )
    connection.commit()