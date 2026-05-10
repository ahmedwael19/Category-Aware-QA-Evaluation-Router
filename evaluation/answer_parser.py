"""LLM answer normalization.

Single parser used across all systems and evaluation stages.
Normalizes raw LLM text to one of: "yes", "no", "reject", or "invalid".
"""

VALID_ANSWERS = {"yes", "no"}
REJECT_ANSWER = "reject"
INVALID_ANSWER = "invalid"


def parse_llm_answer(raw_text):
    """Normalize LLM response to yes/no/reject/invalid."""
    t = raw_text.strip().lower().rstrip(".")
    if t in VALID_ANSWERS | {REJECT_ANSWER}:
        return t
    if "reject" in t and "yes" not in t and "no" not in t:
        return REJECT_ANSWER
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t and "yes" not in t:
        return "no"
    return INVALID_ANSWER
