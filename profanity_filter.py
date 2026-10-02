import re

# ---------------------------------------------------------------
# BLOCKLIST - add or remove words freely. Keep everything lowercase.
# ---------------------------------------------------------------
ENGLISH = [
    "fuck", "fucker", "motherfucker", "shit", "bullshit", "shitty",
    "asshole", "ass", "dumbass", "jackass", "bitch", "bastard",
    "dick", "dickhead", "cock", "pussy", "cunt", "slut", "whore",
    "piss", "prick", "twat", "wanker", "douche", "douchebag",
    "scumbag", "moron", "idiot", "stupid", "scum", "crap",
]

HINGLISH = [
    "madarchod", "maderchod", "behenchod", "bhenchod", "bhosdike",
    "bhosadike", "bhosdi", "bhosda", "chutiya", "chutiye", "chutia",
    "gandu", "gaandu", "gaand", "harami", "haramkhor", "haramzada",
    "kamina", "kamine", "kutta", "kutte", "kutti", "randi", "raand",
    "lund", "lauda", "lode", "loda", "jhatu", "jhaatu", "bhadwa",
    "bhadwe", "suar", "saale", "bakchod", "chod", "chodu", "tatti",
]

BLOCKLIST = sorted(set(ENGLISH + HINGLISH))

# Common letter/number/symbol swaps used to dodge filters: f*ck, sh1t, @ss
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a",
                      "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})

# Match whole words only (so "class" or "assignment" are NOT flagged),
# allowing common endings like -s, -ing, -ed, -er
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in BLOCKLIST) + r")(?:s|es|ed|er|ers|ing|y)?\b"
)


def _variants(text: str):
    t = text.lower()
    leet = t.translate(LEET)
    # "fuuuuck" -> "fuck"
    collapsed = re.sub(r"(.)\1{2,}", r"\1", leet)
    # "f*ck", "sh.t" style: replace * and . inside words with nothing is risky,
    # so we only handle the symbol swaps above.
    return {t, leet, collapsed}


def check_profanity(text: str) -> dict:
    matched = set()
    for variant in _variants(text or ""):
        for m in _PATTERN.finditer(variant):
            matched.add(m.group(1))
    return {"flagged": bool(matched), "matched_terms": sorted(matched)}