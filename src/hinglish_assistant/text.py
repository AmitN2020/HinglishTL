from __future__ import annotations

import re
import unicodedata


def normalize_text(value: str) -> str:
    """Normalize text consistently for model inputs and metric comparisons."""
    value = unicodedata.normalize("NFC", value)
    return re.sub(r"\s+", " ", value).strip()


def validate_roman_hinglish(value: str) -> str:
    value = normalize_text(value)
    if not value:
        raise ValueError("Enter some Roman-script Hinglish text.")
    if len(value) > 500:
        raise ValueError("Keep an input to 500 characters or fewer.")
    return value
