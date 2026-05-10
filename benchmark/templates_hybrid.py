"""HYBRID template-based prompt generation.

Combines a symbolic condition (time, channel, tag, count) with a semantic
judgment in a single prompt.
"""

HYBRID_Q = [
    "If the customer waited more than {n} minutes, did the agent acknowledge the delay?",
    "For responses taking over {n} minutes, was there an apology?",
    "When the wait exceeded {n} minutes, did the agent address it?",
    "For {ch} conversations, was the opening appropriate for the channel?",
    "If this came via {ch}, was the communication style suitable?",
    "On {ch}, did the agent follow the expected etiquette?",
    "If the conversation took more than {n} exchanges, did the agent remain patient?",
    "For conversations exceeding {n} turns, was quality maintained throughout?",
    "If the ticket is marked '{tag}', was it handled with appropriate urgency?",
    "For '{tag}' tickets, was the response quality adequate?",
    "Did the agent both respond within {n} minutes and address the core issue?",
    "Was the {ch} response both timely and genuinely helpful?",
    "If tagged '{tag}' and over {n} turns, was urgency maintained?",
]

HYBRID_I = [
    "Check if the agent apologized when response exceeded {n} minutes.",
    "Evaluate acknowledgment of delay for waits over {n} minutes.",
    "Assess {ch}-appropriate communication style.",
    "Verify that {ch} best practices were followed.",
    "Detect quality degradation in conversations exceeding {n} turns.",
    "Verify '{tag}' tickets received appropriate urgency.",
    "Assess handling appropriateness for '{tag}' cases.",
    "Evaluate both response speed ({n} minute threshold) and helpfulness.",
    "Check if a long wait ({n}+ minutes) was compensated by extra care.",
    "Detect whether {ch} etiquette was maintained under pressure.",
    "Flag '{tag}' tickets where urgency was not matched by quality.",
]

CHANNELS = ["chat", "email", "phone", "twitter", "whatsapp", "web form"]
TAGS = ["urgent", "vip", "complaint", "escalated", "sensitive", "legal", "billing"]
NUMBERS = [3, 5, 10, 15, 20, 30, 45, 60]

def generate_hybrid_prompts(apply_noise, pick_noise_level, pick_style, random_module, count=600):
    """Generate template-based HYBRID prompts.

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
        pool = HYBRID_Q if style == "question" else HYBRID_I
        template = random_module.choice(pool)
        prompt = template.format(
            n=random_module.choice(NUMBERS),
            ch=random_module.choice(CHANNELS),
            tag=random_module.choice(TAGS),
        )
        noise = pick_noise_level()
        prompt = apply_noise(prompt, noise)
        prompts.append({
            "prompt": prompt, "category": "HYBRID",
            "subcategory": "hybrid", "style": style,
            "noise": noise, "params": None,
        })
    return prompts
