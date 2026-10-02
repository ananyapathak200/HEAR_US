import re

# Keyword override: if any of these appear, the complaint is forced to Critical,
# whatever the ML model says. Patterns are regexes on lowercase text.
CRITICAL_PATTERNS = [
    r"\bthreat\w*",                 # threat, threaten, threatened, threatening
    r"\bunsafe\b",
    r"\bassault\w*",
    r"\bragg(?:ing|ed)\b",
    r"\bharass\w*",
    r"\bemergency\b",
    r"\bsexual\w*",
    r"\bmolest\w*",
    r"\brap(?:e|ed|ist)\b",
    r"\bblackmail\w*",
    r"\bweapon\w*",
    r"\bkill\w*",
    r"\bsuicid\w*",
    r"\b(?:end|ending|take|taking)\s+(?:my|his|her|their)\s+(?:own\s+)?life\b",
    r"\bend\s+everything\b",
    r"\b(?:harm|hurt|kill)(?:ing)?\s+(?:myself|himself|herself)\b",
    r"\bunconscious\b",
    r"\bnot\s+breathing\b",
    r"\bnot\s+responding\b",
    r"\bfaint(?:ed|ing)\b",
    r"\belectric(?:al)?\s+shock\w*",
    r"\bgetting\s+shocks?\b",
    r"\bgas\s+leak\w*",
    r"\b(?:fire\s+in|on\s+fire|caught\s+fire|there\s+is\s+a\s+fire)\b",
    r"\bbleeding\b",
    r"\binjur(?:ed|y)\b",
    r"\bbeat(?:en)?\s+up\b",
    r"\bphysically\s+(?:abused|assaulted|attacked)\b",
    r"\bin\s+danger\b",
    r"\binappropriately\b",
    r"\bforcefully\b",
    r"\bfumes\b",
    r"\bpanic\s+attack\b",
    r"\bmedical\s+(?:help|attention|emergency)\b",
    r"\blocked\s+(?:me|him|her|them)\b",
    r"\b(?:hit|hitting|punch\w*|slapp?(?:ed|ing))\s+(?:me|him|her|them)\b",
    r"\bothers\s+hit\b",
    r"\bnearly\s+fell\b",
    r"\bsmoke\b",
    r"\b(?:knife|gun|pistol|blade)\b",
    r"\bself[- ]?harm\w*",
    r"\bcutting\s+(?:myself|himself|herself)\b",
    r"\bnobody\s+would\s+miss\b",
]

_COMPILED = [re.compile(p) for p in CRITICAL_PATTERNS]


def find_critical_keywords(text: str):
    t = (text or "").lower()
    hits = []
    for rx in _COMPILED:
        m = rx.search(t)
        if m:
            hits.append(m.group(0).strip())
    return hits