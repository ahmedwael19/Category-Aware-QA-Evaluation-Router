"""Deterministic noise injection for synthetic benchmark generation.

Applies typos, informal language, grammar drops, punctuation and case
mutations at configurable intensity levels to simulate realistic QA prompt
quality variation.
"""

import re as _re_module

_TYPO_MAP = {
    "a": ["s", "q"], "e": ["r", "w"], "i": ["o", "u"], "o": ["p", "i"],
    "s": ["a", "d"], "t": ["r", "y"], "n": ["m", "b"], "r": ["e", "t"],
    "l": ["k", ";"], "d": ["s", "f"], "c": ["x", "v"], "m": ["n", ","],
}

_WORD_TYPOS = {
    "the": ["teh", "hte", ""], "agent": ["agnet", "agen", "agant"],
    "customer": ["custoemr", "custmer", "cusotmer"],
    "response": ["respnse", "reponse", "resposne"],
    "whether": ["wether", "wheather"], "appropriate": ["approriate", "apropriate"],
    "conversation": ["converstion", "convesation"],
    "message": ["messge", "mesage"], "evaluate": ["evalaute", "evalute"],
    "acknowledge": ["acknowlege", "aknowledge"],
}

_INFORMAL = {
    "please": ["plz", "pls"], "you": ["u"], "because": ["cuz", "bc"],
    "want to": ["wanna"], "going to": ["gonna"], "information": ["info"],
    "without": ["w/o"], "within": ["w/in"], "approximately": ["~", "approx"],
}

_GRAMMAR_DROPS = [
    (r"\bDid the agent\b", "Did agent"), (r"\bdid the agent\b", "did agent"),
    (r"\bWas the\b", "Was"), (r"\bwas the\b", "was"),
    (r"\bif the\b", "if"), (r"\bwhether the\b", "whether"),
    (r"\bCheck if the\b", "Check if"), (r"\bDetect if the\b", "Detect if"),
]

def _char_typo(text, p, random_module):
    if random_module.random() > p:
        return text
    chars = list(text)
    for i, c in enumerate(chars):
        if c.lower() in _TYPO_MAP and random_module.random() < 0.10:
            chars[i] = random_module.choice(_TYPO_MAP[c.lower()])
    return "".join(chars)

def _word_typo(text, p, random_module):
    if random_module.random() > p:
        return text
    for w, typos in _WORD_TYPOS.items():
        if w in text.lower() and random_module.random() < 0.20:
            text = text.replace(w, random_module.choice(typos), 1)
    return text

def _informal(text, p, random_module):
    if random_module.random() > p:
        return text
    for formal, subs in _INFORMAL.items():
        if formal in text.lower() and random_module.random() < 0.25:
            text = text.replace(formal, random_module.choice(subs))
    return text

def _grammar_drop(text, p, random_module):
    if random_module.random() > p:
        return text
    for pat, repl in _GRAMMAR_DROPS:
        if random_module.random() < 0.20:
            text = _re_module.sub(pat, repl, text, count=1)
    return text

def _punct_mess(text, p, random_module):
    if random_module.random() > p:
        return text
    ch = random_module.choice(["rm", "dbl", "ell"])
    if ch == "rm":
        text = text.rstrip("?.!")
    elif ch == "dbl" and text.endswith("?"):
        text += "?"
    elif ch == "ell":
        text = text.rstrip("?.!") + "..."
    return text

def _case_mess(text, p, random_module):
    if random_module.random() > p or not text:
        return text
    if random_module.random() < 0.6:
        return text[0].lower() + text[1:]
    return text.lower()

def apply_noise(text, intensity="medium", random_module=None):
    """Apply noise at given intensity. Returns modified text.

    Parameters
    ----------
    text : str
        The prompt text to corrupt.
    intensity : str
        One of "none", "light", "medium", "heavy".
    random_module : module
        The ``random`` module (or compatible object) for deterministic control.
    """
    if intensity == "none":
        return text
    m = {"light": 0.5, "medium": 1.0, "heavy": 2.0}[intensity]
    text = _char_typo(text, 0.08 * m, random_module)
    text = _word_typo(text, 0.12 * m, random_module)
    text = _informal(text, 0.10 * m, random_module)
    text = _grammar_drop(text, 0.10 * m, random_module)
    text = _punct_mess(text, 0.15 * m, random_module)
    text = _case_mess(text, 0.10 * m, random_module)
    return text

def pick_noise_level(random_module):
    """60% none, 25% light, 10% medium, 5% heavy."""
    r = random_module.random()
    if r < 0.60:
        return "none"
    elif r < 0.85:
        return "light"
    elif r < 0.95:
        return "medium"
    return "heavy"

def pick_style(random_module):
    """Return 'question' or 'instruction' with 50/50 probability."""
    return "question" if random_module.random() < 0.50 else "instruction"
