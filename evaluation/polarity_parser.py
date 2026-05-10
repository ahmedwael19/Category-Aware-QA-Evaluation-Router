"""Strict prompt-polarity parser for symbolic TIME and COUNT prompts.

A prompt is mechanically parseable iff:
  1. it contains a polarity keyword from the whitelist below,
  2. the numeric threshold from params appears literally in the prompt text,
  3. polarity is determined by exactly one of {under, over, equal}.

Prompts that rely on external configuration ("deadline", "SLA", "window",
"threshold met", "promptness", "timely") or on subjective judgement ("too long",
"drawn-out", "streamlined") are not parseable by this module and are excluded
from the end-to-end benchmark by the admission rule.

Used by:
  - evaluation/dataset_builder.py                   (admission gate when building the benchmark)
  - scripts/validation/validate_final_benchmark.py  (post-build audit gate)

The parser is deliberately strict: it returns None rather than guessing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


UNDER_PATTERNS: tuple[str, ...] = (
    r"\bunder\b",
    r"\bwithin\b",
    r"\bw/in\b",
    r"\bwithn\b",  # typo survives
    r"\bbefore\b",
    r"\bless than\b",
    r"\bfewer than\b",
    r"\bshorter than\b",
    r"\bat most\b",
    r"\bno more than\b",
    r"\bup to\b",
    r"\bbelow\b",
    r"\bstayed under\b",
    r"\bstay under\b",
    r"\bin under\b",
    r"\bcame in under\b",
    r"\bunder the\b",
    r"\bshort of\b",
    r"\bwrapped up before\b",
    r"\bbefore hitting\b",
    r"\bbefore \d+\s*(?:messages?|turns?|exchanges?)\b",
    r"\bnot over\b",
    r"\bnot more than\b",
    r"\bnot longer than\b",
    r"\bkept (?:it )?(?:under|below)\b",
    r"\bsub-?\d",  # "sub-90-second"
)

OVER_PATTERNS: tuple[str, ...] = (
    r"\bmore than\b",
    r"\bover\b",
    r"\bpast\b",
    r"\bbeyond\b",
    r"\bafter\b",
    r"\bexceeded\b",
    r"\bexceeds\b",
    r"\bexceeding\b",
    r"\bsurpassed?\b",
    r"\bsurpasses\b",
    r"\bcrossed\b",
    r"\blonger than\b",
    r"\bwider than\b",
    r"\bgreater than\b",
    r"\babove\b",
    r"\bbigger than\b",
    r"\bhigher than\b",
    r"\btook more than\b",
    r"\bwent (?:above|beyond|over|past)\b",
    r"\bgoing beyond\b",
    r"\bin excess of\b",
    r"\bran (?:over|past)\b",
    r"\bout beyond\b",
    r"\bwait(?:ing)? (?:over|past|more than|beyond)\b",
    r"\bkept waiting beyond\b",
    r"\blonger tham\b",  # typo
    r"\bexcceded\b",     # typo
    r"\bwider thsn\b",   # typo
)

EQUAL_PATTERNS: tuple[str, ...] = (
    r"\bexactly\b",
    r"\bprecisely\b",
    r"\bequal to\b",
    r"\bequals\b",
)

# Tokens that indicate the threshold is implicit / external, and therefore the
# prompt is ineligible regardless of whether an UNDER/OVER keyword matched.
BLOCKED_EXTERNAL_TOKENS: tuple[str, ...] = (
    r"\bsla\b",
    r"\bdeadline\b",
    r"\btarget\b",
    r"\bgoal\b",
    r"\bbenchmark\b",
    r"\bstandard\b",
    r"\bstandards\b",
    r"\bwindow (?:was )?honor",
    r"\bexpectation\b",
    r"\bexpected\b",
    r"\btimely\b",
    r"\bpromptness\b",
    r"\bprompt\b",  # "Was the reply prompt?"
    r"\bquick enough\b",
    r"\brapid\b",
    r"\bswift\b",
    r"\bslow\b",
    r"\btoo long\b",
    r"\btoo slow\b",
    r"\btoo (?:many|much)\b",
    r"\bhanging\b",
    r"\blingered\b",
    r"\bdrawn[- ]out\b",
    r"\bstreamlined\b",
    r"\bunnecessarily\b",
    r"\bdelays?\b",
    r"\bdelayed\b",
    r"\blong wait\b",
    r"\bheld up\b",
    r"\blate\b",
    r"\bthreshold (?:was )?met\b",
    r"\btarget (?:was )?achieved\b",
)


@dataclass(frozen=True, slots=True)
class ParseResult:
    polarity: str | None            # "under" | "over" | "equal" | None
    reason: str                     # short tag: "ok", "blocked_external", "no_keyword", "ambiguous", "threshold_not_in_text"
    matched_keywords: tuple[str, ...] = ()


def _count_hits(text: str, patterns: tuple[str, ...]) -> list[str]:
    hits = []
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            hits.append(m.group(0))
    return hits


def _threshold_text_candidates(params: dict) -> list[str]:
    """Produce all surface forms the threshold might appear as in prompt text."""
    if "threshold" not in params:
        return []
    thr = params["threshold"]
    out = [str(thr)]
    try:
        thr_int = int(thr)
        out.append(str(thr_int))
    except Exception:
        pass
    return out


def _threshold_in_text(text: str, params: dict) -> bool:
    cands = _threshold_text_candidates(params)
    if not cands:
        return False
    for cand in cands:
        if re.search(rf"\b{re.escape(cand)}\b", text):
            return True
    return False


def parse_polarity(prompt: str, params: dict | str | None) -> ParseResult:
    """Parse prompt polarity strictly.

    Returns ParseResult with polarity=None if the prompt is not mechanically
    parseable under the strict contract.
    """
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except Exception:
            params = {}
    if not isinstance(params, dict):
        params = {}

    text = (prompt or "").lower()

    # Blocked tokens disqualify the prompt regardless of other matches.
    blocked_hits = _count_hits(text, BLOCKED_EXTERNAL_TOKENS)
    if blocked_hits:
        return ParseResult(None, "blocked_external", tuple(blocked_hits))

    # Threshold must appear literally in text for comparison-style prompts
    # (unless the prompt is for METADATA, which does not use this parser).
    if not _threshold_in_text(text, params):
        return ParseResult(None, "threshold_not_in_text")

    under_hits = _count_hits(text, UNDER_PATTERNS)
    over_hits = _count_hits(text, OVER_PATTERNS)
    equal_hits = _count_hits(text, EQUAL_PATTERNS)

    if equal_hits and not under_hits and not over_hits:
        return ParseResult("equal", "ok", tuple(equal_hits))
    if under_hits and not over_hits:
        return ParseResult("under", "ok", tuple(under_hits))
    if over_hits and not under_hits:
        return ParseResult("over", "ok", tuple(over_hits))
    if under_hits and over_hits:
        return ParseResult(None, "ambiguous", tuple(under_hits + over_hits))
    return ParseResult(None, "no_keyword")


def operator_to_polarity(op: str | None) -> str | None:
    if op is None:
        return None
    s = str(op).strip()
    if s in ("<", "<=", "lt", "lte", "le"):
        return "under"
    if s in (">", ">=", "gt", "gte", "ge"):
        return "over"
    if s in ("==", "=", "eq"):
        return "equal"
    return None


def polarity_to_operator(polarity: str, category: str) -> str:
    """Map parsed polarity to canonical operator string.

    TIME: under -> "<=", over -> ">", equal -> "==".
    COUNT: under -> "<", over -> ">", equal -> "==".

    The asymmetry (<= for time, < for count) matches prompt semantics:
      "under 18 turns" = < 18; "within 18 minutes" = <= 18.
    """
    if polarity == "equal":
        return "=="
    if category == "SYMBOLIC_TIME":
        return "<=" if polarity == "under" else ">"
    if category == "SYMBOLIC_COUNT":
        return "<" if polarity == "under" else ">"
    raise ValueError(f"unknown category for polarity_to_operator: {category}")
