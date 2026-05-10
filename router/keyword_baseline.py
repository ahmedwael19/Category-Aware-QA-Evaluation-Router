"""Keyword/regex-based router baseline.

"""

import re

_TIME_RE = re.compile(
    r'\d+\s*(seconds?|minutes?|hours?|min|sec|hrs?|mins?)'
    r'|response\s+time|sla|turnaround|timel[yi]|prompt\s+respon'
    r'|delay|quick\s+repl|fast\s+enough|wait\s+time'
    r'|within\s+\d|under\s+\d|over\s+\d+\s*(min|sec|hour|hr)',
    re.IGNORECASE,
)

_COUNT_RE = re.compile(
    r'\d+\s*(turns?|messages?|exchanges?|replies?|interactions?|back.and.forth)'
    r'|how\s+many\s+(message|turn|exchange|repl)'
    r'|leng?thy|concise|drawn\s+out|brief\s+conversation|efficient\s+conversation'
    r'|exceed(ed|ing)?\s+\d+',
    re.IGNORECASE,
)

_META_RE = re.compile(
    r'\btag(ged|s)?\b|label(ed|led)?\b|\bchannel\b|internal\s+note'
    r'|source\s+type|marked\s+as|flagg?ed\s+as|\bvia\s+(email|chat|phone)'
    r"|'[a-z_]+'\s*(tag|ticket|label)"
    r"|is\s+(this|the|it)\s+.*\b(email|chat|phone|twitter|whatsapp)\b",
    re.IGNORECASE,
)

_UNSUPPORTED_RE = re.compile(
    r'predict|forecast|churn|lifetime\s+value|purchase\s+history'
    r'|other\s+ticket|previous\s+(ticket|interaction|complaint)'
    r'|customer.s\s+(account|subscription|payment|billing|purchase|refund|login)'
    r'|satisfaction\s+(score|rating)|net\s+promoter|nps'
    r'|come\s+back|follow.up\s+(complaint|issue)'
    r'|compare.*team|agent.*performance.*average'
    r'|across\s+(all\s+)?(ticket|contact|interaction)'
    r'|\b(crm|erp|salesforce|shopify|stripe)\b',
    re.IGNORECASE,
)

_SEMANTIC_WORDS = [
    "empathetic", "empathy", "polite", "professional", "friendly",
    "appropriate", "helpful", "care", "caring", "understanding", "tone",
    "apologize", "apology", "acknowledge", "patient", "attentive",
    "considerate", "compassion", "courtesy", "respectful", "warm",
    "genuinely", "sincere", "suitable", "adequat", "quality",
]


def _has_semantic_indicator(text):
    lower = text.lower()
    return any(w in lower for w in _SEMANTIC_WORDS)


def keyword_router(text):
    """Classify a prompt using regex patterns. Returns category string."""
    has_time = bool(_TIME_RE.search(text))
    has_count = bool(_COUNT_RE.search(text))
    has_meta = bool(_META_RE.search(text))
    has_unsupported = bool(_UNSUPPORTED_RE.search(text))
    has_symbolic = has_time or has_count or has_meta
    has_semantic = _has_semantic_indicator(text)

    if has_unsupported and not has_symbolic:
        return "UNSUPPORTED"
    if has_symbolic and has_semantic:
        return "HYBRID"
    if has_time:
        return "SYMBOLIC_TIME"
    if has_count:
        return "SYMBOLIC_COUNT"
    if has_meta:
        return "SYMBOLIC_METADATA"
    return "SEMANTIC"
