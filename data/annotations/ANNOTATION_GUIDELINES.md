# Annotation Guidelines — QA Prompt Taxonomy Validation

**Task**: Classify 120 QA evaluation prompts into one of 6 categories.
**Time estimate**: ~30-45 minutes.
**Deadline**: March 12, 2026.

---

## Categories

| Category | What it means | Example |
|----------|--------------|---------|
| **SYMBOLIC_TIME** | Can be answered by comparing timestamps or durations. Has an explicit or implicit time threshold. | "Did the agent respond within 5 minutes?" |
| **SYMBOLIC_METADATA** | Can be answered by checking a tag, channel, or field value. Exact-match lookup. | "Is the ticket tagged 'escalated'?" |
| **SYMBOLIC_COUNT** | Can be answered by counting messages, turns, or exchanges. | "Did the conversation exceed 10 turns?" |
| **SEMANTIC** | Requires subjective judgment about quality, tone, empathy, or communication. No deterministic answer exists. | "Was the agent empathetic?" |
| **HYBRID** | Contains BOTH a verifiable condition (time/count/tag/channel) AND a subjective judgment. Both parts are needed. | "If the wait exceeded 5 min, did the agent apologize?" |
| **UNSUPPORTED** | References data not available in a single conversation transcript (cross-ticket history, CRM data, future predictions). | "Has this customer escalated before?" |

---

## Decision Procedure

Apply these checks in order:

1. **Does the prompt reference data unavailable in a single conversation?** (other tickets, CRM, customer history, future predictions) → **UNSUPPORTED**

2. **Does the prompt require a computation** (time comparison, tag check, counting)? If NO → **SEMANTIC**

3. **Does the prompt ALSO require a subjective judgment?** If YES → **HYBRID**. If NO → continue.

4. **What computation?**
   - Time/duration/SLA → **SYMBOLIC_TIME**
   - Tag/channel/field lookup → **SYMBOLIC_METADATA**
   - Counting messages/turns → **SYMBOLIC_COUNT**

---

## Key Boundary Cases

**Vague time references are SEMANTIC, not SYMBOLIC_TIME:**
- "Did the agent respond quickly?" → SEMANTIC (no explicit threshold)
- "Did the agent respond within 5 minutes?" → SYMBOLIC_TIME (explicit threshold)

**HYBRID requires BOTH halves:**
- "For chat conversations, was the tone professional?" → HYBRID (channel check + tone judgment)
- "Was the tone professional?" → SEMANTIC (no symbolic condition)

**The removal test for HYBRID:** If you remove the symbolic condition and the evaluation fundamentally changes, it's HYBRID. If removing it doesn't change the evaluation, it's SEMANTIC.

---

## How to Fill the CSV

For each prompt in `annotation_study.csv`:

1. **annotator_category**: One of: `SYMBOLIC_TIME`, `SYMBOLIC_METADATA`, `SYMBOLIC_COUNT`, `SEMANTIC`, `HYBRID`, `UNSUPPORTED`
2. **annotator_confidence**: `high`, `medium`, or `low`
3. **notes**: Optional — only if the prompt is ambiguous or you want to explain your reasoning

---

## What Data Is Available in a Conversation

When deciding if a prompt is answerable, assume the evaluator has access to:
- Message text, speaker (agent/customer), timestamps
- Whether each message is public or internal note
- Channel (per message): chat, email, phone, etc.
- Ticket tags (list of strings)
- Ticket source type
- Total resolution time

The evaluator does NOT have:
- Other tickets from the same customer
- CRM or billing system data
- Post-conversation satisfaction scores
- Future behavior predictions
- Agent performance benchmarks
