"""Scenario-conditioned LLM generation for SEMANTIC, UNSUPPORTED, and HYBRID categories.

Contains the scenario dimensions, category definitions, prompt templates,
and batch generation functions for all LLM-generated prompt categories.
"""

import json as _json
import time as _time

SCENARIOS = [
    "billing error", "flight delay", "account locked", "lost password",
    "subscription cancellation", "product defect", "shipping delay",
    "refund request", "service outage", "data breach notification",
    "wrong item received", "payment failure", "feature not working",
    "account suspension", "overcharge complaint",
]

STAGES = [
    "greeting", "initial troubleshooting", "escalation request",
    "resolution delivery", "closing", "follow-up",
]

BEHAVIORS = [
    "apologizing", "validating feelings", "active listening",
    "acknowledging frustration", "offering alternatives",
    "explaining next steps", "checking understanding",
    "personalizing the response", "showing patience",
    "taking ownership of the issue",
]

STRUCTURES = ["absolute", "conditional"]

SEMANTIC_CATS = {
    "SEMANTIC_EMPATHY": {
        "desc": "Whether the agent showed understanding of the customer's situation, acknowledged emotions, made the customer feel heard.",
        "forbidden": "empathy, empathetic, feelings, emotions, emotional, understand, understanding, compassion, sympathetic, sympathy",
    },
    "SEMANTIC_TONE": {
        "desc": "Whether the agent's communication style was appropriate — friendly, professional, not rude or cold, matching the situation.",
        "forbidden": "tone, tonal, professional, professionalism, friendly, polite, politeness, appropriate, rude, rudeness, manner, demeanor",
    },
    "SEMANTIC_SOLUTION": {
        "desc": "Whether the agent actually helped solve the problem or gave generic canned responses without real assistance.",
        "forbidden": "solution, solve, resolved, resolution, fix, fixed, answer, answered, help, helped, assistance",
    },
    "SEMANTIC_GREETING": {
        "desc": "Whether the agent properly opened the conversation with a suitable introduction and acknowledgment of the customer.",
        "forbidden": "greeting, greet, greeted, hello, welcome, welcomed, introduction, introduce, opening, opener",
    },
    "SEMANTIC_CLOSING": {
        "desc": "Whether the agent properly ended the conversation, offered further assistance, and wrapped up appropriately.",
        "forbidden": "closing, close, closed, goodbye, farewell, ending, wrap, wrapping, sign-off, signoff",
    },
    "SEMANTIC_COMPREHENSION": {
        "desc": "Whether the agent understood what the customer was asking, grasped the core issue, and did not misinterpret.",
        "forbidden": "comprehension, comprehend, understand, understood, understanding, grasp, grasped, misunderstand",
    },
}

PROMPT_TEMPLATE = """You generate QA evaluation prompts for a customer service quality assessment system.

    RULES:
    1. DO NOT use these forbidden words: {forbidden}
    2. Generate prompts that real QA managers would write. Include realistic imperfections in ~50%:
       - Occasional typos, informal language, missing articles, inconsistent punctuation
    3. Each prompt MUST be unique in structure and meaning
    4. Mix: ~50% question style ("Did the agent...?"), ~50% instruction style ("Detect whether...")
    5. Vary lengths: ~30% short (<15 words), ~50% medium (15-40 words), ~20% long (40+ words)

    CATEGORY: {category}
    DESCRIPTION: {desc}

    CONDITIONING (use these to create diverse prompts):
    - Scenario context: {scenario}
    - Interaction stage: {stage}
    - Observable behavior: {behavior}
    - Structure: {structure}

    Generate exactly {n} diverse prompts conditioned on the above dimensions.
    Return JSON: {{"prompts": ["...", ...]}}"""

UNSUPPORTED_LLM_PROMPT = """You generate QA evaluation prompts that are IMPOSSIBLE to answer
    from a single customer support conversation transcript.

    RULES:
    1. Each prompt MUST reference data that is NOT available in a single conversation transcript.
       Unavailable data includes: cross-ticket history, post-conversation outcomes (CSAT, NPS),
       external system data (CRM, billing, subscription databases), future predictions,
       agent performance comparisons, customer lifetime metrics.
    2. DO NOT generate prompts that can be answered from the conversation messages, timestamps,
       tags, or channel information — those are SUPPORTED categories.
    3. Generate prompts that real QA managers might naively write, not realizing the data is unavailable.
    4. Mix: ~50% question style ("Did the customer...?"), ~50% instruction style ("Check if...")
    5. Each prompt MUST be unique in meaning and structure.
    6. DO NOT use these obvious giveaway words: "predict", "forecast", "CRM", "external", "database"
    7. Vary lengths: ~30% short (<15 words), ~50% medium (15-40 words), ~20% long (40+ words)

    CATEGORY: UNSUPPORTED — prompts that reference data unavailable at evaluation time.

    CONDITIONING (use these to create diverse prompts):
    - Unavailability type: {unavail_type}
    - Domain context: {domain}

    Generate exactly {n} diverse prompts.
    Return JSON: {{"prompts": ["...", ...]}}"""

HYBRID_LLM_PROMPT = """You generate QA evaluation prompts for a customer service quality assessment system.

    CRITICAL CONSTRAINT — EVERY prompt MUST contain TWO halves:
    1) A strict numerical or metadata CONDITION (e.g., "If the wait was over 10 minutes...",
       "For VIP tickets...", "When the conversation exceeded 5 turns...", "On email...")
    2) A subjective SEMANTIC JUDGMENT (e.g., "...did the agent sound empathetic?",
       "...was the tone professional?", "...did they adequately address the concern?")

    The condition can be:
    - Conditional: "If {{condition}}, {{judgment}}?"
    - Composite: "{{judgment}} AND {{condition}}?"
    - Temporal: "When {{condition}}, {{judgment}}?"
    - Filtered: "For tickets where {{condition}}, {{judgment}}?"

    If a prompt lacks EITHER the symbolic condition OR the semantic judgment, it is INVALID.

    RULES:
    1. Each prompt MUST combine a verifiable condition with a subjective evaluation
    2. DO NOT generate pure semantic prompts (no condition) or pure symbolic prompts (no judgment)
    3. Mix: ~50% question style, ~50% instruction style
    4. Each prompt MUST be unique in meaning and structure
    5. DO NOT use forbidden words: "hybrid", "combined", "both", "dual"
    6. Vary lengths: ~30% short, ~50% medium, ~20% long

    CONDITIONING:
    - Symbolic dimension: {symbolic_dim}
    - Semantic dimension: {semantic_dim}
    - Domain context: {domain}

    Generate exactly {n} diverse prompts.
    Return JSON: {{"prompts": ["...", ...]}}"""

UNAVAIL_TYPES = [
    "cross-ticket data (customer's other interactions)",
    "post-conversation outcomes (satisfaction score, follow-up)",
    "external system records (billing, subscription, purchase history)",
    "predictive/hypothetical (future behavior, churn likelihood)",
    "comparative data (agent vs team performance, benchmarks)",
    "customer profile data (lifetime value, account tier, preferences)",
]

UNSUPPORTED_DOMAINS = [
    "e-commerce refund", "SaaS subscription", "airline complaint",
    "banking dispute", "telecom service issue", "insurance claim",
    "food delivery complaint", "hotel booking issue",
]

SYMBOLIC_DIMS = [
    "response time threshold (minutes)", "message count threshold (turns)",
    "channel type (email/chat/phone)", "ticket tag (urgent/vip/escalated)",
    "internal notes presence", "resolution time",
]

SEMANTIC_DIMS = [
    "empathy and emotional acknowledgment", "professional tone and courtesy",
    "solution quality and helpfulness", "greeting appropriateness",
    "closing and follow-up offer", "comprehension of customer issue",
]

HYBRID_DOMAINS = [
    "billing dispute", "technical troubleshooting", "subscription cancellation",
    "shipping complaint", "account security issue", "product return",
    "service outage", "upgrade request",
]

def generate_semantic_batch(llm_client, cat_name, n=25, scenario=None,
                            stage=None, behavior=None, structure=None,
                            random_module=None):
    """Generate a batch of SEMANTIC prompts conditioned on scenario dimensions.

    Parameters
    ----------
    llm_client : openai.OpenAI
        Configured OpenAI client.
    cat_name : str
        Key into ``SEMANTIC_CATS``.
    n : int
        Number of prompts per batch.
    scenario, stage, behavior, structure : str or None
        Conditioning dimensions; randomly chosen if None.
    random_module : module
        The ``random`` module for deterministic control.

    Returns
    -------
    list[str]
        Generated prompt strings.
    """
    cat = SEMANTIC_CATS[cat_name]

    _scenario = scenario or random_module.choice(SCENARIOS)
    _stage = stage or random_module.choice(STAGES)
    _behavior = behavior or random_module.choice(BEHAVIORS)
    _structure = structure or random_module.choice(STRUCTURES)

    prompt_text = PROMPT_TEMPLATE.format(
        forbidden=cat["forbidden"], category=cat_name,
        desc=cat["desc"], n=n,
        scenario=_scenario, stage=_stage,
        behavior=_behavior, structure=_structure,
    )

    for attempt in range(3):
        try:
            resp = llm_client.chat.completions.create(
                model="gpt-5.2",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You generate diverse QA evaluation prompts. Return only valid JSON with a 'prompts' key containing an array of strings."},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=1.0,
                max_completion_tokens=4000,
            )
            result = _json.loads(resp.choices[0].message.content)
            prompts = result.get("prompts", [])
            if isinstance(prompts, list) and len(prompts) > 0:
                return prompts
        except Exception as e:
            print(f"      Attempt {attempt+1} failed: {e}")
            _time.sleep(2)
    return []

def generate_unsupported_llm_batch(llm_client, n=25, unavail_type=None,
                                   domain=None, random_module=None):
    """Generate a batch of LLM-based UNSUPPORTED prompts.

    Parameters
    ----------
    llm_client : openai.OpenAI
        Configured OpenAI client.
    n : int
        Number of prompts per batch.
    unavail_type : str or None
        Unavailability type; randomly chosen if None.
    domain : str or None
        Domain context; randomly chosen if None.
    random_module : module
        The ``random`` module for deterministic control.

    Returns
    -------
    list[str]
    """
    _utype = unavail_type or random_module.choice(UNAVAIL_TYPES)
    _domain = domain or random_module.choice(UNSUPPORTED_DOMAINS)

    prompt_text = UNSUPPORTED_LLM_PROMPT.format(
        unavail_type=_utype, domain=_domain, n=n,
    )
    for attempt in range(3):
        try:
            resp = llm_client.chat.completions.create(
                model="gpt-5.2",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You generate diverse QA evaluation prompts that reference unavailable data. Return only valid JSON with a 'prompts' key containing an array of strings."},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=1.0,
                max_completion_tokens=4000,
            )
            result = _json.loads(resp.choices[0].message.content)
            prompts = result.get("prompts", [])
            if isinstance(prompts, list) and len(prompts) > 0:
                return prompts
        except Exception as e:
            print(f"      Attempt {attempt+1} failed: {e}")
            _time.sleep(2)
    return []

def generate_hybrid_llm_batch(llm_client, n=25, symbolic_dim=None,
                              semantic_dim=None, domain=None,
                              random_module=None):
    """Generate a batch of LLM-based HYBRID prompts.

    Parameters
    ----------
    llm_client : openai.OpenAI
        Configured OpenAI client.
    n : int
        Number of prompts per batch.
    symbolic_dim : str or None
        Symbolic conditioning dimension; randomly chosen if None.
    semantic_dim : str or None
        Semantic conditioning dimension; randomly chosen if None.
    domain : str or None
        Domain context; randomly chosen if None.
    random_module : module
        The ``random`` module for deterministic control.

    Returns
    -------
    list[str]
    """
    _sym = symbolic_dim or random_module.choice(SYMBOLIC_DIMS)
    _sem = semantic_dim or random_module.choice(SEMANTIC_DIMS)
    _dom = domain or random_module.choice(HYBRID_DOMAINS)

    prompt_text = HYBRID_LLM_PROMPT.format(
        symbolic_dim=_sym, semantic_dim=_sem, domain=_dom, n=n,
    )
    for attempt in range(3):
        try:
            resp = llm_client.chat.completions.create(
                model="gpt-5.2",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You generate QA evaluation prompts that MUST combine a symbolic condition with a semantic judgment. Every prompt needs both halves. Return only valid JSON with a 'prompts' key."},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=1.0,
                max_completion_tokens=4000,
            )
            result = _json.loads(resp.choices[0].message.content)
            prompts = result.get("prompts", [])
            if isinstance(prompts, list) and len(prompts) > 0:
                return prompts
        except Exception as e:
            print(f"      Attempt {attempt+1} failed: {e}")
            _time.sleep(2)
    return []
