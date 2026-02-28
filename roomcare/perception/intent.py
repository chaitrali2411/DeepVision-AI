"""Intent extraction from transcribed text (care-related events)."""
from __future__ import annotations

import re
from typing import Optional

from roomcare.config import EVENT_LABELS


# Keywords that suggest care intents (no raw audio stored; only intent metadata)
INTENT_PATTERNS = {
    "injection_event": re.compile(
        r"\b(inject|injection|syringe|medication|dose|IV push)\b",
        re.I,
    ),
    "iv_change_event": re.compile(
        r"\b(IV|bag|replace|change|hang|drip|line|infusion)\b",
        re.I,
    ),
}


class IntentExtractor:
    """Extract care-related intent from transcribed text."""

    def __init__(self):
        self.patterns = INTENT_PATTERNS

    def extract(self, text: str) -> str | None:
        """Return injection_event, iv_change_event, or None if no clear intent."""
        if not (text and text.strip()):
            return None
        text = text.strip()
        for label, pat in self.patterns.items():
            if pat.search(text):
                return label
        return None
