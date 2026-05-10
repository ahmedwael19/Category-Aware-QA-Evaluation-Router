"""SYMBOLIC_TIME template-based prompt generation.

Contains all TIME template arrays (explicit, moderate, implicit) for both
question and instruction styles, and the generation function.
"""

import json as _json

TQ_EXPLICIT = [
    "Did the agent respond within {n} {u}?",
    "Was the first reply sent in under {n} {u}?",
    "Was there a response within {n} {u}?",
    "Did the customer receive a reply within {n} {u}?",
    "Was the agent's first message within {n} {u} of the query?",
    "Did the support rep answer within the {n} {u} window?",
    "Was the customer contacted back within {n} {u}?",
    "Did the representative respond in less than {n} {u}?",
    "Was the response delivered before {n} {u} passed?",
    "Did the first reply take less than {n} {u}?",
    "Was there an answer from the agent within {n} {u}?",
    "Did the agent meet the {n} {u} response target?",
    "Was a reply provided within {n} {u} of the initial message?",
    "Did the team member get back within {n} {u}?",
    "Was the query addressed within {n} {u}?",
    "Did the agent's response arrive in under {n} {u}?",
    "Was the {n} {u} response window honored?",
    "Did the support agent reply inside {n} {u}?",
    "Was the customer's message acknowledged within {n} {u}?",
    "Did the agent beat the {n} {u} deadline?",
]

TQ_MODERATE = [
    "Did the customer have to wait more than {n} {u} for a reply?",
    "Was the initial response to the customer sent within {n} {u}?",
    "Did they get back to the customer within {n} {u}?",
    "Was the customer kept waiting beyond {n} {u}?",
    "Did the reply arrive before the {n} {u} mark?",
    "Did the customer end up waiting over {n} {u}?",
    "Was the gap between the query and response under {n} {u}?",
    "Did the response come before the {n} {u} limit?",
    "Was the time to first response shorter than {n} {u}?",
    "Did the agent take longer than {n} {u} to get back?",
    "Was the customer left hanging for more than {n} {u}?",
    "Did the turnaround on the first reply stay under {n} {u}?",
    "Was the delay between customer message and agent reply under {n} {u}?",
    "Did the customer get an answer before {n} {u} elapsed?",
    "Was there more than a {n} {u} gap before the response?",
    "Did the agent's reply come through within the {n} {u} cutoff?",
    "Was the response time longer or shorter than {n} {u}?",
    "Did the customer need to wait past {n} {u} for acknowledgment?",
    "Was the first contact from the agent made within {n} {u}?",
    "Did the queue time exceed {n} {u}?",
]

TQ_IMPLICIT = [
    "Was the reply timely?",
    "Did the agent respond fast enough?",
    "Was there an unreasonable delay before the first response?",
    "Did the customer wait too long?",
    "Was response speed acceptable?",
    "Was there a quick turnaround on the response?",
    "Did the agent take too long to respond?",
    "Was the first reply prompt?",
    "Did the customer experience a long wait?",
    "Was the response time reasonable?",
    "Did the agent reply in a timely manner?",
    "Was there a noticeable delay?",
    "Did the customer get a fast response?",
    "Was the wait time within acceptable limits?",
    "Did the reply come promptly?",
]

TI_EXPLICIT = [
    "Check if the reply came within {n} {u}.",
    "Verify the response was sent under {n} {u}.",
    "Determine if first response exceeded {n} {u}.",
    "Assess whether the {n} {u} threshold was met.",
    "Confirm the reply was delivered within {n} {u}.",
    "Validate that the agent responded inside {n} {u}.",
    "Measure whether the first response took less than {n} {u}.",
    "Ensure the {n} {u} target was achieved.",
    "Examine if the response was sent before the {n} {u} limit.",
    "Test whether the reply met the {n} {u} requirement.",
    "Audit the response against the {n} {u} standard.",
    "Inspect if the first reply came under {n} {u}.",
    "Ascertain whether {n} {u} was exceeded.",
    "Gauge if the response fell within {n} {u}.",
    "Report whether the {n} {u} window was breached.",
]

TI_MODERATE = [
    "Detect if there was a delay over {n} {u}.",
    "Flag if customer waited more than {n} {u}.",
    "Evaluate response speed against the {n} {u} target.",
    "Identify cases where {n} {u} deadline was missed.",
    "Measure whether the {n} {u} SLA was breached.",
    "Spot instances where the agent exceeded {n} {u}.",
    "Note if the response gap was wider than {n} {u}.",
    "Highlight cases where {n} {u} was not met.",
    "Record whether the customer waited beyond {n} {u}.",
    "Track if the turnaround time surpassed {n} {u}.",
    "Pinpoint responses that came after {n} {u}.",
    "Monitor whether the {n} {u} benchmark was hit.",
    "Watch for reply times over {n} {u}.",
    "Catalog instances of {n} {u} threshold violations.",
    "Log responses that exceeded the {n} {u} mark.",
]

TI_IMPLICIT = [
    "Evaluate response timeliness.",
    "Detect excessive wait before first reply.",
    "Flag slow initial responses.",
    "Assess whether the response was prompt.",
    "Identify delays in agent response.",
    "Check for slow turnaround times.",
    "Flag cases with delayed first contact.",
    "Spot unreasonable response delays.",
    "Note any slow agent replies.",
    "Detect sluggish response patterns.",
    "Monitor for late initial responses.",
    "Identify poor response speed.",
    "Flag tardy agent replies.",
    "Assess first-reply promptness.",
    "Detect response time issues.",
]

UNITS = ["seconds", "minutes", "hours", "sec", "min", "hrs"]
NUMBERS = list(range(1, 31)) + [45, 60, 90, 120, 180, 300]

def generate_time_prompts(apply_noise, pick_noise_level, pick_style, random_module, count=600):
    """Generate SYMBOLIC_TIME prompts.

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
        List of prompt dicts with keys: prompt, category, subcategory,
        style, noise, params.
    """
    pool_map = {
        ("question", "explicit"): TQ_EXPLICIT, ("question", "moderate"): TQ_MODERATE,
        ("question", "implicit"): TQ_IMPLICIT, ("instruction", "explicit"): TI_EXPLICIT,
        ("instruction", "moderate"): TI_MODERATE, ("instruction", "implicit"): TI_IMPLICIT,
    }

    prompts = []
    for _ in range(count):
        style = pick_style()
        level = random_module.choice(["explicit", "moderate", "implicit"])
        n = random_module.choice(NUMBERS)
        u = random_module.choice(UNITS)

        _pool = pool_map[(style, level)]
        template = random_module.choice(_pool)
        prompt = template.format(n=n, u=u) if "{n}" in template else template
        noise = pick_noise_level()
        prompt = apply_noise(prompt, noise)

        prompts.append({
            "prompt": prompt, "category": "SYMBOLIC_TIME",
            "subcategory": f"time_{level}", "style": style,
            "noise": noise, "params": _json.dumps({"threshold": n, "unit": u}),
        })
    return prompts
