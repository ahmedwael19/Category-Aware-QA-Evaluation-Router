"""SYMBOLIC_COUNT template-based prompt generation.

Contains all COUNT template arrays (explicit, moderate, implicit) for both
question and instruction styles, and the generation function.
"""

import json as _json

CQ_EXPLICIT = [
    "Did the conversation have more than {n} turns?",
    "Were there fewer than {n} messages?",
    "Did the agent send more than {n} replies?",
    "Did the exchange exceed {n} messages?",
    "Were more than {n} back-and-forth exchanges needed?",
    "Did the total number of messages go above {n}?",
    "Was the conversation longer than {n} exchanges?",
    "Did the agent write more than {n} responses?",
    "Were {n} or more turns required to address this?",
    "Did the thread contain at least {n} messages?",
    "Was the message count above {n}?",
    "Did more than {n} replies get sent by the agent?",
    "Were {n}+ messages exchanged in this conversation?",
    "Did the customer and agent exchange over {n} messages?",
    "Was the turn count greater than {n}?",
]

CQ_MODERATE = [
    "Was it resolved in under {n} messages?",
    "Did it take more than {n} exchanges to resolve?",
    "Was the thread longer than {n} turns?",
    "Did the conversation stay under {n} messages?",
    "Was the issue handled in fewer than {n} replies?",
    "Did the back-and-forth go past {n} exchanges?",
    "Was the conversation wrapped up before hitting {n} messages?",
    "Did the interaction require over {n} turns?",
    "Was the exchange completed within {n} messages?",
    "Did solving the issue take more than {n} replies?",
    "Was {n} messages enough or did it go longer?",
    "Did the agent need more than {n} attempts to address this?",
    "Was the conversation shorter or longer than {n} turns?",
    "Did it go over {n} exchanges before reaching resolution?",
    "Was the total interaction under {n} back-and-forths?",
]

CQ_IMPLICIT = [
    "Was this a quick resolution?",
    "Did the conversation drag on unnecessarily?",
    "Was the handling efficient?",
    "Was there excessive back-and-forth?",
    "Was the issue resolved without unnecessary exchanges?",
    "Did it take too many messages to resolve?",
    "Was the conversation concise?",
    "Did the interaction go on longer than needed?",
    "Was the exchange brief or extended?",
    "Did the agent resolve it efficiently?",
    "Was the conversation drawn out?",
    "Did the agent keep the exchange short?",
    "Was the back-and-forth minimal?",
    "Did the conversation overshoot a reasonable length?",
    "Was the resolution delivered without excess exchanges?",
]

CI_EXPLICIT = [
    "Check if more than {n} turns occurred.",
    "Verify message count is under {n}.",
    "Detect conversations exceeding {n} exchanges.",
    "Identify if more than {n} replies were sent.",
    "Count the number of agent responses and compare to {n}.",
    "Confirm the thread had fewer than {n} messages.",
    "Assess whether the turn count exceeded {n}.",
    "Determine if the exchange surpassed {n} messages.",
    "Validate that the conversation stayed under {n} turns.",
    "Measure if the total messages crossed {n}.",
    "Report whether {n} exchanges were exceeded.",
    "Audit the conversation length against a {n} message limit.",
    "Examine if the reply count went above {n}.",
    "Test whether fewer than {n} turns were used.",
    "Inspect the thread for more than {n} exchanges.",
]

CI_MODERATE = [
    "Evaluate if resolved within {n} turns.",
    "Flag if the conversation exceeded {n} messages.",
    "Detect lengthy interactions over {n} exchanges.",
    "Assess whether {n} messages was sufficient.",
    "Highlight conversations running past {n} turns.",
    "Monitor for exchanges going beyond {n} messages.",
    "Spot threads that took over {n} replies.",
    "Note interactions requiring more than {n} turns.",
    "Track conversations exceeding the {n} message mark.",
    "Identify threads where {n} exchanges were not enough.",
    "Record cases where more than {n} replies were needed.",
    "Watch for turn counts above {n}.",
    "Pinpoint interactions exceeding {n} back-and-forths.",
    "Log conversations that passed {n} messages.",
    "Catalog threads over {n} exchanges long.",
]

CI_IMPLICIT = [
    "Assess conversation efficiency.",
    "Identify quick resolutions.",
    "Flag drawn-out conversations.",
    "Detect unnecessarily long interactions.",
    "Evaluate whether the exchange was concise.",
    "Spot inefficient conversation patterns.",
    "Note excessive back-and-forth.",
    "Monitor for verbose interactions.",
    "Flag conversations that could have been shorter.",
    "Detect resolution delays from excessive exchanges.",
    "Identify bloated threads.",
    "Assess whether the interaction was streamlined.",
    "Highlight unnecessarily lengthy exchanges.",
    "Track conversation efficiency patterns.",
    "Flag cases of prolonged back-and-forth.",
]

NUMBERS = list(range(2, 20)) + [25, 30, 50]

def generate_count_prompts(apply_noise, pick_noise_level, pick_style, random_module, count=600):
    """Generate SYMBOLIC_COUNT prompts.

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
    pool_map = {
        ("question", "explicit"): CQ_EXPLICIT, ("question", "moderate"): CQ_MODERATE,
        ("question", "implicit"): CQ_IMPLICIT, ("instruction", "explicit"): CI_EXPLICIT,
        ("instruction", "moderate"): CI_MODERATE, ("instruction", "implicit"): CI_IMPLICIT,
    }

    prompts = []
    for _ in range(count):
        style = pick_style()
        level = random_module.choice(["explicit", "moderate", "implicit"])
        n = random_module.choice(NUMBERS)

        _pool = pool_map[(style, level)]
        template = random_module.choice(_pool)
        prompt = template.format(n=n) if "{n}" in template else template
        noise = pick_noise_level()
        prompt = apply_noise(prompt, noise)

        prompts.append({
            "prompt": prompt, "category": "SYMBOLIC_COUNT",
            "subcategory": f"count_{level}", "style": style,
            "noise": noise, "params": _json.dumps({"threshold": n}),
        })
    return prompts
