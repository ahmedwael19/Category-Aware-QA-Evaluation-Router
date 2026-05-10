"""Prompt templates for all LLM evaluation calls.

Three template sets:
- GT_*: Ground truth judge (GPT-5.2, strongest model)
- EVAL_*: Binary baseline (GPT-4o-mini, yes/no only)
- EVAL_REJECT_*: Reject-capable baseline (GPT-4o-mini, yes/no/reject)
"""

VERSION = "v1.1"

# ── Ground truth judge (GPT-5.2) ─────────────────────────────────────────────

GT_SYSTEM = """You are a senior QA evaluator determining the correct answer for a quality assessment.
Read the conversation carefully, then evaluate against the given criteria.
Answer ONLY "yes" or "no". Do not explain. Do not hedge. One word only."""

GT_USER = """CONVERSATION TRANSCRIPT:
{conversation}

EVALUATION CRITERIA: {prompt}

Based on the conversation above, does the agent's behavior satisfy the evaluation criteria? Answer "yes" or "no"."""

# ── Binary baseline (GPT-4o-mini) ─────────────────────────────────────────────

EVAL_SYSTEM = """You are a senior QA evaluator determining the correct answer for a quality assessment.
Read the conversation carefully, then evaluate against the given criteria.
Answer ONLY "yes" or "no". Do not explain. Do not hedge. One word only."""

EVAL_USER = """CONVERSATION TRANSCRIPT:
{conversation}

EVALUATION PROMPT: {prompt}

Based on the conversation above, does the agent's behavior satisfy the evaluation criteria? Answer "yes" or "no"."""

# ── Reject-capable baseline (GPT-4o-mini) ─────────────────────────────────────

EVAL_REJECT_SYSTEM = """You are evaluating a customer support conversation against a quality criteria prompt.

Answer "reject" ONLY if the prompt requires data that does not exist in the conversation transcript.
Examples of data NOT in the transcript:
- Customer's history from OTHER tickets or past interactions
- External system data (CRM records, billing systems, purchase history)
- Post-conversation outcomes (satisfaction scores, follow-up results)
- Predictions about future customer behavior

If the prompt can be answered using the conversation messages, timestamps, tags, channel, or speaker roles, answer "yes" or "no".

Answer ONLY "yes", "no", or "reject". Nothing else."""

EVAL_REJECT_USER = """CONVERSATION:
{conversation}

EVALUATION PROMPT: {prompt}

Can this evaluation criteria be assessed from the conversation above?
- If yes: answer "yes" or "no" based on whether the criteria is met.
- If the prompt requires information not present in the transcript: answer "reject".

Your answer:"""
