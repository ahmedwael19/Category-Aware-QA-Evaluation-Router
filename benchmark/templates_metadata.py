"""SYMBOLIC_METADATA template-based prompt generation.

Contains all METADATA template arrays for tags, channels, and internal notes
in both question and instruction styles.
"""

import json as _json

TAGS = [
    "escalated", "urgent", "vip", "refund", "complaint", "billing",
    "technical", "feedback", "resolved", "pending", "high-priority",
    "follow-up", "spam", "duplicate", "sensitive", "legal",
    "retention", "onboarding", "bug", "feature-request",
]

CHANNELS = ["email", "chat", "phone", "social media", "web form", "twitter", "whatsapp", "SMS"]

TAG_Q = [
    "Does the ticket have the '{v}' label?",
    "Is '{v}' applied to this ticket?",
    "Is this marked as '{v}'?",
    "Was it tagged '{v}'?",
    "Has this been flagged as '{v}'?",
    "Is the '{v}' tag present on this ticket?",
    "Is this an '{v}' case?",
]

TAG_I = [
    "Check if tagged as '{v}'.",
    "Confirm '{v}' is present.",
    "Detect if '{v}' label exists.",
    "Verify ticket has '{v}' status.",
    "Identify '{v}' cases.",
    "Flag tickets marked '{v}'.",
]

CHAN_Q = [
    "Did this come through {v}?",
    "Is this a {v} conversation?",
    "Was the channel {v}?",
    "Is this from {v}?",
    "Was this submitted via {v}?",
    "Is this a {v} ticket?",
]

CHAN_I = [
    "Check if channel is {v}.",
    "Detect {v} conversations.",
    "Identify if source was {v}.",
    "Verify this came via {v}.",
    "Filter for {v} tickets.",
]

NOTES_Q = [
    "Did the agent add an internal note?",
    "Are there internal notes in this conversation?",
    "Did they leave a note for the team?",
    "Any private comments added?",
    "Were behind-the-scenes notes added?",
    "Any non-public messages from the agent?",
]

NOTES_I = [
    "Check for internal comments.",
    "Detect if agent added private notes.",
    "Identify internal documentation.",
    "Verify internal notes exist.",
    "Flag conversations with internal notes.",
]

def generate_metadata_prompts(apply_noise, pick_noise_level, pick_style, random_module, count=600):
    """Generate SYMBOLIC_METADATA prompts.

    Parameters
    ----------
    apply_noise : callable
        ``apply_noise(text, intensity)`` from noise module.
    pick_noise_level : callable
        ``pick_noise_level()`` from noise module.
    pick_style : callable
        ``pick_style()`` from noise module.
    random_module : module
        The ``random`` module for deterministic control.
    count : int
        Number of prompts to generate.

    Returns
    -------
    list[dict]
    """
    prompts = []
    for _ in range(count):
        style = pick_style()
        # Weights reflect structural prevalence in typical QA schemas (Tags > Channels > Notes)
        subtype = random_module.choices(["tag", "channel", "notes"], weights=[0.50, 0.35, 0.15])[0]

        if subtype == "tag":
            v = random_module.choice(TAGS)
            pool = TAG_Q if style == "question" else TAG_I
            prompt = random_module.choice(pool).format(v=v)
            params = {"subtype": "tag", "value": v}
        elif subtype == "channel":
            v = random_module.choice(CHANNELS)
            pool = CHAN_Q if style == "question" else CHAN_I
            prompt = random_module.choice(pool).format(v=v)
            params = {"subtype": "channel", "value": v}
        else:
            pool = NOTES_Q if style == "question" else NOTES_I
            prompt = random_module.choice(pool)
            params = {"subtype": "notes", "value": None}

        noise = pick_noise_level()
        prompt = apply_noise(prompt, noise)
        prompts.append({
            "prompt": prompt, "category": "SYMBOLIC_METADATA",
            "subcategory": f"meta_{subtype}", "style": style,
            "noise": noise, "params": _json.dumps(params),
        })
    return prompts
