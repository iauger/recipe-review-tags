# src/utils/text_cleaning.py

from typing import Union
import re
from pyspark.sql import functions as F
from pyspark.sql import Column

def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and removing punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text) # Remove punctuation but keep hyphens
    text = re.sub(r"\s+", " ", text).strip() # Remove extra spaces
    return text

def normalize_text_spark(col: Union[str, Column]) -> Column:
    c = F.col(col) if isinstance(col, str) else col
    c = F.coalesce(c, F.lit(""))

    return F.trim(
        F.regexp_replace(
            F.regexp_replace(
                F.lower(c),
                r"[^a-z0-9\s-]",
                " "
            ),
            r"\s+",
            " "
        )
    )