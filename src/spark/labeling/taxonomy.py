from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# NOTE:
# - Keep tag ids stable once you start generating labels.
# - Descriptions are for humans / reporting.
# - hypotheses are for zero-shot (NLI-style) classifiers.
#   They should read naturally when inserted into:
#       "This review expresses that the recipe was {hypothesis}."
#   or similar templates used in zero_shot.py.


@dataclass(frozen=True)
class Tag:
    id: str
    label: str
    description: str
    group: str
    hypotheses: List[str]
    polarity: str  # "positive", "negative", or "neutral"
    examples: Optional[List[str]] = None


def get_taxonomy(version: str = "v1") -> Dict[str, Tag]:
    """
    Returns the tag taxonomy used for labeling.
    version: enables future iteration without breaking old experiments.
    """
    if version != "v1":
        raise ValueError(f"Unknown taxonomy version: {version}")

    tags: List[Tag] = [
        # -------------------------
        # Taste / flavor
        # -------------------------
        Tag(
            id="too_salty",
            label="Too salty",
            group="taste",
            description="Reviewer says the dish was overly salty or needed less salt.",
            hypotheses=[
                "too salty",
                "too much salt",
            ],
            examples=[
                "way too salty",
                "next time I'll cut the salt in half",
            ],
            polarity="negative",
        ),
        Tag(
            id="too_sweet",
            label="Too sweet",
            group="taste",
            description="Reviewer says the dish was overly sweet or needed less sugar/sweetener.",
            hypotheses=[
                "too sweet",
                "too much sugar",
            ],
            examples=[
                "a bit too sweet for my taste",
                "I'll reduce the sugar next time",
            ],
            polarity="negative",
        ),
        Tag(
            id="too_acidic",
            label="Too acidic",
            group="taste",
            description="Reviewer says the dish was overly acidic/sour/tangy or needed less acid (lemon/vinegar).",
            hypotheses=[
                "too sour",
                "too acidic",
            ],
            examples=[
                "a bit too sour for my taste",
                "I'll reduce the vinegar next time",
                "I'll use less lemon juice next time",
            ],
            polarity="negative",
        ),
        Tag(
            id="bland_lacks_flavor",
            label="Bland / lacks flavor",
            group="taste",
            description="Reviewer says it was bland or needed more seasoning/flavor.",
            hypotheses=[
                "bland",
                "needed more seasoning",
            ],
            examples=[
                "kind of bland",
                "needed more spices",
            ],
            polarity="negative",
        ),
        Tag(
            id="too_spicy",
            label="Too spicy",
            group="taste",
            description="Reviewer says the dish was too spicy/hot or needed less heat.",
            hypotheses=[
                "too spicy",
                "too hot",
            ],
            examples=[
                "this was really spicy",
                "too hot for the kids",
                "too much spice",
            ],
            polarity="negative",
        ),
        Tag(
            id="delicious_tasty",
            label="Delicious / tasty",
            group="taste",
            description="Reviewer explicitly praises taste or flavor (delicious, amazing, great flavor).",
            hypotheses=[
                "delicious",
                "tasted great",
            ],
            examples=[
                "so delicious",
                "amazing flavor",
                "this was really tasty",
            ],
            polarity="positive",
        ),

        # -------------------------
        # Texture / doneness
        # -------------------------
        Tag(
            id="dry",
            label="Dry",
            group="texture",
            description="Reviewer says the dish came out dry or overcooked.",
            hypotheses=[
                "dry",
                "overcooked",
            ],
            examples=[
                "came out dry",
                "a little overcooked",
            ],
            polarity="negative",
        ),
        Tag(
            id="mushy_soggy",
            label="Mushy / soggy",
            group="texture",
            description="Reviewer says the result was mushy, soggy, watery, or lacked structure.",
            hypotheses=[
                "soggy",
                "too watery",
            ],
            examples=[
                "too soggy",
                "turned out watery",
            ],
            polarity="negative",
        ),
        Tag(
            id="crispy_crunchy",
            label="Crispy / crunchy",
            group="texture",
            description="Reviewer praises crispness/crunchiness or a crunchy texture.",
            hypotheses=[
                "crispy",
                "crunchy",
            ],
            examples=[
                "nice and crispy",
                "great crunch",
            ],
            polarity="positive",
        ),
        Tag(
            id="moist_tender",
            label="Moist / tender",
            group="texture",
            description="Reviewer praises moist, tender, juicy, or soft texture.",
            hypotheses=[
                "moist and tender",
                "juicy",
            ],
            examples=[
                "super moist",
                "tender and juicy",
                "nice and soft",
            ],
            polarity="positive",
        ),

        # -------------------------
        # Process / difficulty
        # -------------------------
        Tag(
            id="easy_quick",
            label="Easy / quick",
            group="process",
            description="Reviewer says it was easy, simple, quick, or straightforward.",
            hypotheses=[
                "easy to make",
                "quick and easy",
            ],
            examples=[
                "so easy",
                "quick weeknight dinner",
                "finished in 30 minutes",
            ],
            polarity="positive",
        ),
        Tag(
            id="time_consuming_complex",
            label="Time-consuming / complex",
            group="process",
            description="Reviewer says it took a long time, was complicated, or instructions were unclear.",
            hypotheses=[
                "time consuming",
                "instructions were confusing",
            ],
            examples=[
                "took forever",
                "instructions were confusing",
                "took much longer than expected",
            ],
            polarity="negative",
        ),

        # -------------------------
        # Ingredients / adjustments
        # -------------------------
        Tag(
            id="substitution_modification",
            label="Substitution / modification",
            group="ingredients",
            description="Reviewer describes substitutions or intentional changes to the recipe (swap/replace/add).",
            hypotheses=[
                "I substituted an ingredient",
                "I changed the recipe",
            ],
            examples=[
                "I used Greek yogurt instead of sour cream",
                "I doubled the garlic",
                "I used gluten-free flour instead of all-purpose flour",
            ],  
            polarity="neutral",
        ),
        Tag(
            id="ingredient_issue",
            label="Ingredient issue",
            group="ingredients",
            description="Reviewer reports a true ingredient/ratio problem (amounts off, missing ingredient, ingredient didn't work).",
            hypotheses=[
                "the ingredient amounts were off",
                "missing an ingredient",
            ],
            examples=[
                "needed more liquid",
                "too much garlic",
                "too much fat",
            ],
            polarity="negative",
        ),

        # -------------------------
        # Outcome / intent
        # -------------------------
        Tag(
            id="would_make_again",
            label="Would make again",
            group="outcome",
            description="Reviewer indicates they would make it again, keep it, or recommend it.",
            hypotheses=[
                "will make again",
                "a keeper",
            ],
            examples=[
                "this is a keeper",
                "will definitely make again",
            ],
            polarity="positive",
        ),
        Tag(
            id="would_not_make_again",
            label="Would NOT make again",
            group="outcome",
            description="Reviewer indicates they would not make it again or would not recommend it.",
            hypotheses=[
                "won't make again",
                "would not recommend",
            ],
            examples=[
                "won't be making this again",
                "would not recommend",
            ],
            polarity="negative",
        ),
        Tag(
            id="family_hit",
            label="Family hit / crowd pleaser",
            group="outcome",
            description="Reviewer says others loved it (family loved it, a hit, crowd pleaser).",
            hypotheses=[
                "a hit",
                "everyone loved it",
            ],
            examples=[
                "a big hit at our house",
                "everyone loved it",
                "the kids loved it",
            ],
            polarity="positive",
        ),
    ]

    return {t.id: t for t in tags}


def get_tag_ids(version: str = "v1") -> List[str]:
    return list(get_taxonomy(version=version).keys())


def get_zero_shot_candidate_labels(version: str = "v1") -> List[str]:
    """
    Candidate labels to pass to a zero-shot classifier.
    """
    return get_tag_ids(version=version)


def get_label_to_hypotheses(version: str = "v1") -> Dict[str, List[str]]:
    """
    Mapping used by zero-shot code when you want multiple hypotheses per label.
    """
    tax = get_taxonomy(version=version)
    return {tid: tax[tid].hypotheses for tid in tax}


def get_groups(version: str = "v1") -> Dict[str, List[str]]:
    """
    Convenience: group -> tag_ids
    """
    tax = get_taxonomy(version=version)
    groups: Dict[str, List[str]] = {}
    for t in tax.values():
        groups.setdefault(t.group, []).append(t.id)
    return groups

def get_tag_polarity(version: str = "v1") -> Dict[str, str]:
    tax = get_taxonomy(version=version)
    return {tid: tax[tid].polarity for tid in tax}