"""UNSUPPORTED template-based prompt generation.

Parameterized templates across four unavailability subtypes: cross-ticket,
post-conversation, external system, and predictive/hypothetical.
"""

METRICS = ["purchase", "refund", "return", "login", "support ticket",
           "subscription", "billing", "usage", "complaint", "escalation"]

SYSTEMS = ["CRM", "Shopify", "Stripe", "the app", "external database",
           "third-party system", "social media", "partner portal", "ERP", "Salesforce"]

OUTCOMES = ["churn", "escalate", "renew", "cancel", "upgrade",
            "downgrade", "leave negative feedback", "request a callback"]

PERIODS = ["week", "month", "quarter", "30 days", "next billing cycle", "90 days"]

DATA_TYPES = ["account balance", "subscription tier", "payment method",
              "contact preferences", "loyalty points", "credit score"]

HISTORY_TYPES = ["history", "record", "pattern", "trend", "log"]

N_TICKETS = [2, 3, 5, 10, 15, 20, 50]

UNSUPPORTED_Q_TEMPLATES = [
    # Cross-ticket (parameterized)
    "What is this customer's average {metric} {history} across all tickets?",
    "How many times has this customer had {metric} issues in previous tickets?",
    "Did this customer have unresolved {metric} problems in the last {n} tickets?",
    "Is this a repeat {metric} complaint from the same customer?",
    "Has the customer filed more than {n} {metric} requests before?",
    "What was the {metric} outcome of their last {n} interactions?",
    "How often does this customer {outcome}?",
    "Across the last {n} tickets, what is the {metric} trend?",
    # Post-conversation (parameterized)
    "Did the customer rate this interaction positively after {period}?",
    "Was there a follow-up {metric} complaint after this ticket was closed?",
    "Did the customer come back with the same {metric} issue within {period}?",
    "What satisfaction score was given for this {metric} ticket?",
    "Did the customer {outcome} after this conversation ended?",
    # External system (parameterized)
    "What is the customer's {metric} {history} in {system}?",
    "Is this customer's {data_type} available from {system}?",
    "What does {system} show for this customer's {data_type}?",
    "Can we verify the customer's {data_type} against {system}?",
    "What is the customer's {data_type} according to {system} records?",
    # Predictive / hypothetical (parameterized)
    "Will this customer {outcome} in the next {period}?",
    "Is this interaction likely to lead to a {metric} within {period}?",
    "Would the customer recommend the service after this {metric} interaction?",
    "What is the probability the customer will {outcome}?",
    "Based on this conversation, will the customer's {data_type} change?",
]

UNSUPPORTED_I_TEMPLATES = [
    # Cross-ticket
    "Evaluate based on the customer's past {metric} interactions across {n} tickets.",
    "Check the customer's complete {metric} {history} from {system}.",
    "Cross-reference this {metric} issue with {system} data.",
    "Aggregate the customer's {metric} pattern from the last {n} contacts.",
    # External system
    "Verify the customer's {data_type} in {system}.",
    "Assess based on the customer's {metric} records in {system}.",
    "Pull the customer's {data_type} from {system} before evaluating.",
    "Compare this agent's handling against the customer's {history} in {system}.",
    # Predictive
    "Predict {outcome} likelihood based on this {metric} interaction.",
    "Forecast the customer's {metric} trend over the next {period}.",
    "Estimate the probability that this customer will {outcome} within {period}.",
    "Determine if this {metric} interaction changes the customer's {data_type}.",
    "Calculate the impact of this conversation on {metric} over {period}.",
    "Assess whether the customer is likely to {outcome} after this exchange.",
]

def generate_unsupported_prompts(apply_noise, pick_noise_level, pick_style, random_module, count=600):
    """Generate template-based UNSUPPORTED prompts.

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
        pool = UNSUPPORTED_Q_TEMPLATES if style == "question" else UNSUPPORTED_I_TEMPLATES
        template = random_module.choice(pool)
        prompt = template.format(
            metric=random_module.choice(METRICS),
            history=random_module.choice(HISTORY_TYPES),
            system=random_module.choice(SYSTEMS),
            outcome=random_module.choice(OUTCOMES),
            period=random_module.choice(PERIODS),
            data_type=random_module.choice(DATA_TYPES),
            n=random_module.choice(N_TICKETS),
        )
        noise = pick_noise_level()
        prompt = apply_noise(prompt, noise)
        prompts.append({
            "prompt": prompt, "category": "UNSUPPORTED",
            "subcategory": "unsupported", "style": style,
            "noise": noise, "params": None,
        })
    return prompts
