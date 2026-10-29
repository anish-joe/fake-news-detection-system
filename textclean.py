import re

_dateline = re.compile(
    r"^\s*(?:[A-Z][A-Za-z .,'/-]{0,60}\s*)?" r"\(\s*Reuters\s*\)\s*[-–—:]*\s*"
)

_trailing_patterns = [
    re.compile(
        r"(?i)\s+(?:photo|image)\s+by\s+[^.!?\n]{1,160}"
        r"\b(?:Getty Images|Associated Press|AP Photo)\b"
        r"[^.!?\n]*[.!?]?\s*$"
    ),
    re.compile(
        r"(?i)\s+(?:image via|featured image)\s+[^.!?\n]{1,160}" r"(?:[.!?]\s*|$)"
    ),
    re.compile(r"(?i)\s+(?:read more|share this|advertisement)" r"\s*[.!|]*\s*$"),
]

_patterns = [
    re.compile(r"https?://\S+|www\.\S+", re.I),
    re.compile(r"pic\.twitter\.com/\S+", re.I),
    re.compile(r"@\w+"),
    re.compile(r"[^a-z0-9\s'.!?-]"),
    re.compile(r"\s+"),
]


def clean(text):
    text = str(text).strip()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = _dateline.sub("", text)

    for pattern in _trailing_patterns:
        text = pattern.sub("", text)

    text = text.lower()

    for pattern in _patterns:
        text = pattern.sub(" ", text)

    return text.strip()
