# src/utils/text_cleaning.py

import re

def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and removing punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text) # Remove punctuation but keep hyphens
    text = re.sub(r"\s+", " ", text).strip() # Remove extra spaces
    return text