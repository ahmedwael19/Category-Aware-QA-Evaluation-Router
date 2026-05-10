"""Deterministic evaluation functions for SYMBOLIC prompt categories.

Each function takes a Conversation plus parameters and returns "yes" or "no".
Same input always produces the same output — zero variance by construction.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    role: str             # "agent" or "customer"
    text: str
    timestamp: datetime
    is_public: bool = True
    channel: str = "chat"


@dataclass
class Conversation:
    messages: list        # list[Message]
    tags: list = field(default_factory=list)
    channel: str = "chat"
    resolution_time_minutes: float | None = None


_CANONICAL_OP = {
    "<": "<", "lt": "<",
    "<=": "<=", "lte": "<=", "le": "<=",
    ">": ">", "gt": ">",
    ">=": ">=", "gte": ">=", "ge": ">=",
    "==": "==", "eq": "==",
}


def _normalise_operator(operator, fn_name):
    if operator is None or operator not in _CANONICAL_OP:
        raise ValueError(f"{fn_name} requires an explicit operator; got {operator!r}")
    return _CANONICAL_OP[operator]


def first_response_time_minutes(conv):
    """Minutes between the first public customer message and the next agent reply."""
    customer_ts = None
    for m in conv.messages:
        if m.role == "customer" and m.is_public:
            customer_ts = m.timestamp
            break
    if customer_ts is None:
        return None
    for m in conv.messages:
        if m.role == "agent" and m.is_public and m.timestamp > customer_ts:
            return (m.timestamp - customer_ts).total_seconds() / 60.0
    return None


def check_response_time(conv, threshold, operator):
    """Whether the first agent response time satisfies `threshold operator`.

    `operator` is required (one of <, <=, >, >=, ==). Silent defaults mask
    polarity bugs, so callers must pass the operator explicitly.
    """
    op = _normalise_operator(operator, "check_response_time")
    rt = first_response_time_minutes(conv)
    if rt is None:
        return "no"
    ok = {
        "<":  rt < threshold,
        "<=": rt <= threshold,
        ">":  rt > threshold,
        ">=": rt >= threshold,
        "==": abs(rt - threshold) < 0.01,
    }[op]
    return "yes" if ok else "no"


def check_tag_present(conv, tag_name):
    return "yes" if tag_name.lower() in [t.lower() for t in conv.tags] else "no"


def check_channel(conv, expected_channel):
    return "yes" if conv.channel.lower() == expected_channel.lower() else "no"


def check_internal_notes_exist(conv):
    return "yes" if any(not m.is_public for m in conv.messages) else "no"


def count_messages(conv, public_only=True):
    if public_only:
        return sum(1 for m in conv.messages if m.is_public)
    return len(conv.messages)


def check_message_count(conv, threshold, operator):
    """Whether the public message count satisfies `threshold operator`."""
    op = _normalise_operator(operator, "check_message_count")
    count = count_messages(conv)
    ok = {
        "<":  count < threshold,
        "<=": count <= threshold,
        ">":  count > threshold,
        ">=": count >= threshold,
        "==": count == threshold,
    }[op]
    return "yes" if ok else "no"
