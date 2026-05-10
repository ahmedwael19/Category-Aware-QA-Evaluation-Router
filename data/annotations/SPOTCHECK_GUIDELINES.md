# Annotation Guidelines — Semantic / Hybrid Ground-Truth Spot-Check

**Task**: For each of 70 (prompt, conversation) pairs, decide whether the agent's behavior in the conversation satisfies the evaluation prompt. Answer **yes** or **no**.
**Time estimate**: ~60–90 minutes.

This is a **blinded** validation. The annotator does not see any model's answer. Annotators should not discuss items until both CSVs are returned.

---

## What You Are Judging

You are playing the role of a senior QA reviewer. For each row, read the conversation and then read the evaluation prompt. Decide: **based only on what is in the conversation, does the agent's behavior satisfy the criterion stated in the prompt?**

- `yes` — on balance, the agent satisfies the criterion.
- `no` — on balance, the agent does not satisfy the criterion, OR the required behavior is absent from the conversation.

Each prompt falls into one of two categories (already indicated in the `category` column):

| Category | What it means | Example |
|----------|--------------|---------|
| **SEMANTIC** | A subjective judgment about tone, empathy, clarity, professionalism, communication quality. No deterministic check — use your reviewer judgment. | "Was the agent empathetic when the customer expressed frustration?" |
| **HYBRID** | A subjective judgment **conditional on** a symbolic precondition (channel, tag, time, count). Both halves matter. | "If the ticket is tagged 'urgent', did the agent open with a calm and reassuring tone?" |

There are 40 HYBRID and 30 SEMANTIC prompts, presented in shuffled order.

---

## Decision Procedure

**Step 1 — Read the full conversation.** Note the channel and tags in the header, then read the message sequence top to bottom. Mentally place yourself at the end of the conversation.

**Step 2 — Read the evaluation prompt.**

**Step 3 — For SEMANTIC prompts**, judge the agent's behavior against the stated criterion. Ask: "Would a careful reviewer agree the agent clearly demonstrated this?" If yes → **yes**. If they clearly did not, or the criterion's target behavior is absent from the conversation → **no**.

**Step 4 — For HYBRID prompts**, work in two sub-steps:

  - **4a.** Check whether the symbolic precondition holds in this conversation. (E.g., is the channel `chat`? Is the tag `refund` present? Did the first response take more than 5 minutes?)
  - **4b.** If the precondition is **met**, judge the subjective part exactly as you would a SEMANTIC prompt. Your answer is `yes` if the agent satisfied the subjective part, `no` if not.
  - **4b'.** If the precondition is **not met**, the conditional is vacuously satisfied — answer **yes**. (See boundary cases below; this is the standard logical reading.)

---

## Key Boundary Cases

**HYBRID with unmet precondition is `yes`, not `no`:**
- Prompt: "If the wait exceeded 5 min, did the agent apologize?" Conversation: first response in 90 seconds, no apology needed. → **yes** (the condition wasn't triggered, so there was nothing to satisfy).
- Prompt: "For chat conversations, was the tone professional?" Conversation channel is `phone`. → **yes** (condition not applicable here).
- Reasoning: the prompt is a logical conditional. If the "if" part is false, the statement is satisfied by default.
- If the antecedent is ambiguous (e.g., "For long interactions" with no numeric threshold), use your judgment on whether it applies, and add a note.

**"Yes" requires evidence, not absence of counter-evidence:**
- If the prompt asks "Did the agent acknowledge the customer's frustration?" and the customer was never frustrated, answer **no** — the behavior being asked about didn't occur.
- If the customer WAS frustrated and the agent did not acknowledge it → **no**.
- If the agent explicitly acknowledged frustration → **yes**.

**SEMANTIC is about *the agent's* behavior:**
- "Was the tone professional?" is about the agent, not the customer.
- If the customer was rude but the agent stayed professional → **yes**.

**Internal notes count as agent behavior:**
- Lines marked `[INTERNAL NOTE]` are the agent's own notes (not visible to the customer). They are part of the conversation record and can be used as evidence.

**Timestamps matter for HYBRID time-conditioned prompts:**
- All timestamps are in the conversation header and on each message. Compute elapsed time between first customer message and first agent message when needed.

**Multi-turn behavior:**
- Judge the conversation as a whole, not a single turn. The agent may start weakly and recover, or vice versa. Your answer should reflect the overall arc.

---

## How to Fill the CSV

You were sent one of: `semantic_spotcheck_annotator_1.csv` or `semantic_spotcheck_annotator_2.csv`. Both files have the same 70 rows. You will fill in only the file sent to you.

For each row:

1. **`annotator_answer`** — exactly one of `yes` or `no` (lowercase, single word, no punctuation).
2. **`notes`** — optional. Fill this only when:
   - You found the prompt or conversation genuinely ambiguous.
   - You applied the "vacuously `yes`" rule for HYBRID and want to flag it.
   - You noticed the conversation was cut off, malformed, or missing context.
   - Brief is fine — one sentence is enough.

Do **not** edit `id`, `source_idx`, `category`, `prompt`, or `conversation`. Do **not** delete rows. Do **not** reorder rows. Do **not** skip rows — if you genuinely cannot decide, pick the answer you lean toward and leave a note.

Save the file with the **same name** it was sent with and return it by email or the agreed upload link.

---

## What Data Is Available in Each Conversation

Each `conversation` cell is formatted as a short transcript with a header line and then one message per line. You have access to:

- Message text, speaker (`CUSTOMER` / `AGENT`), timestamps.
- Whether each message is public or an `[INTERNAL NOTE]`.
- Channel of the conversation (e.g., `chat`, `email`, `phone`).
- Ticket tags (list of strings in the header).

You do **not** have access to:

- Other tickets from the same customer.
- CRM or billing data.
- Post-conversation satisfaction scores.
- Anything that happened after the last message shown.

If a prompt seems to require data you don't have, judge based only on what is in the conversation and answer `no` if the evidence is absent.

---

## Independence and Blinding

- Work alone. Do not consult the other annotator, this thesis's author, or any LLM while annotating.
- Do not look up the "intended" answer.
- If you are stuck on a single item, leave a note and move on — ambiguous cases are exactly what this study is designed to surface.

Thank you — this validation is essential for the thesis.
