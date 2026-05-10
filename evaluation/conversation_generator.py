"""Synthetic conversation generator with controlled properties.

Generates conversations where timestamps, tags, message counts, and channels
are exactly controlled so that deterministic function ground truth can be
computed.
"""

import json
import random
from datetime import datetime, timedelta

from evaluation.deterministic import Conversation, Message

random.seed(42)

_CUSTOMER_OPENERS = [
    "Hi, I need help with {topic}.",
    "Hello, I'm having an issue with {topic}.",
    "I've been trying to {topic} but it's not working.",
    "Can someone help me with {topic}?",
    "I'm frustrated — {topic} hasn't been resolved.",
    "Quick question about {topic}.",
    "Urgent: {topic} is causing problems for our team.",
    "We've noticed an issue with {topic} recently.",
]
_AGENT_RESPONSES = [
    "I understand, let me look into {topic} for you.",
    "Thank you for reaching out. I'll check on {topic} right away.",
    "I apologize for the inconvenience with {topic}. Let me investigate.",
    "I can help with that. Let me pull up the details on {topic}.",
    "I see the issue with {topic}. Here's what I can do.",
    "Let me review your {topic} situation.",
]
_AGENT_FOLLOWUPS = [
    "I've checked and here's what I found regarding {topic}.",
    "I've applied a fix for the {topic} issue.",
    "Could you try again? I've updated the {topic} settings.",
    "I've escalated the {topic} matter to our specialist team.",
    "The {topic} issue should now be resolved. Please verify.",
]
_CUSTOMER_REPLIES = [
    "Thank you, that helps.",
    "OK, I'll try that.",
    "That's still not working for me.",
    "Could you explain further?",
    "Great, that fixed it!",
    "I appreciate the quick response.",
    "Is there anything else I should know?",
]
_AGENT_CLOSINGS = [
    "Is there anything else I can help with?",
    "Please don't hesitate to reach out if you need more help.",
    "I'll close this ticket. Feel free to reopen if needed.",
    "Glad I could help. Have a great day!",
]
_INTERNAL_NOTES = [
    "Customer seems frustrated. Priority handling.",
    "Checked logs — issue is on our backend.",
    "Escalating to tier 2 support.",
    "Applied workaround. Permanent fix in next release.",
    "Customer confirmed resolution.",
]
_TOPICS = [
    "my account login", "the billing charge", "a shipping delay",
    "the subscription renewal", "password reset", "order cancellation",
    "product returns", "service outage", "the integration setup",
    "data export", "API access", "the mobile app crash",
]
_CHANNELS = ["chat", "email", "phone"]
_TAG_POOL = [
    "billing", "urgent", "vip", "complaint", "escalated",
    "technical", "refund", "feedback", "bug", "feature_request",
]


def generate_conversation(
    n_turns=None,
    response_time_minutes=None,
    channel=None,
    tags=None,
    include_internal_notes=False,
    topic=None,
):
    """Generate a single conversation with controlled properties.

    Args:
        n_turns: total public messages (if None, random 4-12)
        response_time_minutes: minutes between first customer msg and first agent reply
        channel: "chat", "email", "phone" (if None, random)
        tags: list of tags (if None, random 0-3)
        include_internal_notes: whether to add internal notes
        topic: conversation topic (if None, random)
    """
    n = n_turns if n_turns is not None else random.randint(4, 12)
    ch = channel or random.choice(_CHANNELS)
    tgs = tags if tags is not None else random.sample(_TAG_POOL, k=random.randint(0, 3))
    tp = topic or random.choice(_TOPICS)
    rt = response_time_minutes if response_time_minutes is not None else random.uniform(0.5, 30)

    base_time = datetime(2025, 1, 15, 10, 0, 0)
    messages = []

    # First customer message
    messages.append(Message(
        "customer", random.choice(_CUSTOMER_OPENERS).format(topic=tp),
        base_time, True, ch,
    ))

    # First agent reply (controlled response time)
    agent_time = base_time + timedelta(minutes=rt)
    messages.append(Message(
        "agent", random.choice(_AGENT_RESPONSES).format(topic=tp),
        agent_time, True, ch,
    ))

    # Optional internal note
    if include_internal_notes:
        note_time = agent_time + timedelta(seconds=random.randint(10, 60))
        messages.append(Message("agent", random.choice(_INTERNAL_NOTES), note_time, False, ch))

    # Remaining turns (alternating customer/agent)
    current_time = agent_time
    remaining = n - 2
    for i in range(remaining):
        gap = timedelta(minutes=random.uniform(0.5, 5))
        current_time = current_time + gap
        if i % 2 == 0:
            messages.append(Message(
                "customer", random.choice(_CUSTOMER_REPLIES),
                current_time, True, ch,
            ))
        else:
            pool = _AGENT_CLOSINGS if i == remaining - 1 else _AGENT_FOLLOWUPS
            messages.append(Message(
                "agent", random.choice(pool).format(topic=tp),
                current_time, True, ch,
            ))
            if include_internal_notes and random.random() < 0.3:
                note_time = current_time + timedelta(seconds=random.randint(5, 30))
                messages.append(Message("agent", random.choice(_INTERNAL_NOTES), note_time, False, ch))

    resolution = (current_time - base_time).total_seconds() / 60.0
    return Conversation(
        messages=messages, tags=tgs, channel=ch,
        resolution_time_minutes=round(resolution, 1),
    )


# Symbolic-row conversation generator: produces a conversation consistent with
# stored (threshold, unit, operator) params and a chosen scenario.

def _threshold_minutes(params: dict) -> float:
    thr = float(params["threshold"])
    unit = params.get("unit", "minutes")
    if unit in ("seconds", "sec", "secs"):
        return thr / 60.0
    if unit in ("hours", "hrs", "hr"):
        return thr * 60.0
    if unit in ("days", "day"):
        return thr * 1440.0
    return thr


def _pick_response_time(params: dict, scenario: str, operator: str) -> float:
    """Pick an actual first-response time that realises the scenario under the operator.

    scenario controls how far the actual value falls from the threshold:
      clear_yes  -> answer yes, with margin
      clear_no   -> answer no, with margin
      boundary   -> within 10% of the threshold on whichever side still produces
                    the *natural* 'yes' semantics for the operator
      standard   -> pick arbitrarily; result is deterministic from the sampled value
    """
    thr = _threshold_minutes(params)
    # For numeric stability pick positive margins.
    lo = max(0.01, thr * 0.5) if thr > 0 else 0.1
    hi = max(thr * 1.5, thr + 1.0)
    if operator in ("<", "<=", "le", "lt"):
        yes_val = random.uniform(max(0.01, thr * 0.3), max(0.02, thr * 0.8))
        no_val = thr + random.uniform(max(0.5, thr * 0.2), max(1.0, thr * 0.6))
    elif operator in (">", ">=", "gt", "ge"):
        yes_val = thr + random.uniform(max(0.5, thr * 0.2), max(1.0, thr * 0.6))
        no_val = max(0.01, thr - random.uniform(max(0.2, thr * 0.2), max(0.4, thr * 0.6)))
    else:
        yes_val = thr
        no_val = thr
    if scenario == "clear_yes":
        return round(yes_val, 2)
    if scenario == "clear_no":
        return round(no_val, 2)
    if scenario == "boundary":
        # 5-10% from threshold on the yes side
        if operator in ("<", "<=", "le", "lt"):
            return round(max(0.01, thr * 0.95), 2)
        return round(thr + max(0.05, thr * 0.05), 2)
    # fall-back: pick yes
    return round(yes_val, 2)


def _pick_count(params: dict, scenario: str, operator: str) -> int:
    """Pick a concrete message count to realise scenario under operator."""
    thr = int(params["threshold"])
    if operator in ("<", "<=", "le", "lt"):
        yes_val = max(2, thr - random.randint(1, max(1, thr // 3)))
        no_val = thr + random.randint(1, max(1, thr // 3))
    elif operator in (">", ">=", "gt", "ge"):
        yes_val = thr + random.randint(1, max(1, thr // 3))
        no_val = max(2, thr - random.randint(1, max(1, thr // 3)))
    else:
        yes_val = thr
        no_val = thr
    if scenario == "clear_yes":
        return yes_val
    if scenario == "clear_no":
        return no_val
    if scenario == "boundary":
        # choose the yes side, 1 off from threshold
        if operator in ("<", "<=", "le", "lt"):
            return max(2, thr - 1)
        return thr + 1
    return yes_val


def _serialise_conv(conv: Conversation) -> dict:
    return {
        "messages": [
            {
                "role": m.role,
                "text": m.text,
                "timestamp": m.timestamp.isoformat(sep=" "),
                "is_public": bool(m.is_public),
                "channel": m.channel,
            }
            for m in conv.messages
        ],
        "tags": list(conv.tags),
        "channel": conv.channel,
        "resolution_time_minutes": conv.resolution_time_minutes,
    }


def generate_conversation_for_symbolic(row: dict) -> dict:
    """Build a conversation JSON for a SYMBOLIC_TIME/COUNT row with explicit operator.

    Input row shape:
      {"prompt": str, "category": "SYMBOLIC_TIME"|"SYMBOLIC_COUNT",
       "params": {"threshold": ..., "unit": ..., "operator": ...},
       "scenario": "clear_yes"|"clear_no"|"boundary"|"standard"}

    Returns a dict:
      {"json": str,                 # JSON-serialised conversation
       "ground_truth": "yes"|"no",
       "actual_response_time_min": float|None,
       "actual_count": int|None}
    """
    from evaluation.deterministic import (
        check_response_time, check_message_count,
    )
    params = row["params"] if isinstance(row["params"], dict) else json.loads(row["params"])
    scenario = row.get("scenario") or "standard"
    cat = row["category"]
    op = params.get("operator")
    assert op is not None, f"params.operator missing for prompt {row.get('prompt')!r}"

    if cat == "SYMBOLIC_TIME":
        rt = _pick_response_time(params, scenario, op)
        conv = generate_conversation(response_time_minutes=rt)
        thr_min = _threshold_minutes(params)
        gt = check_response_time(conv, thr_min, op)
        return {
            "json": json.dumps(_serialise_conv(conv)),
            "ground_truth": gt,
            "actual_response_time_min": rt,
            "actual_count": None,
        }
    if cat == "SYMBOLIC_COUNT":
        n = _pick_count(params, scenario, op)
        conv = generate_conversation(n_turns=n)
        gt = check_message_count(conv, int(params["threshold"]), op)
        # actual public count (should equal n_turns we asked for; use count_messages)
        from evaluation.deterministic import count_messages
        actual = count_messages(conv, public_only=True)
        return {
            "json": json.dumps(_serialise_conv(conv)),
            "ground_truth": gt,
            "actual_response_time_min": None,
            "actual_count": actual,
        }
    raise ValueError(f"unsupported category: {cat}")
