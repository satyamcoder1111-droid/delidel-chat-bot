"""
Lightweight product catalog query refinement.

The CSV is used as a local index only. We never pass catalog rows to the LLM,
so token usage stays flat even as the catalog grows.
"""

import csv
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path


CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "product_catalog.csv"

ALIASES = {
    "frys": "fries",
    "frice": "fries",
    "frize": "fries",
    "fry": "fries",
    "fries": "french fries",
    "chk": "chicken",
    "chiken": "chicken",
    "chikken": "chicken",
    "log": "leg",
    "qtr": "quarter",
    "nagast": "nuggets",
    "naggets": "nuggets",
    "sousege": "sausage",
    "sosage": "sausage",
    "sawarma": "shawarma",
}

FRIES_BRANDS = {
    "farmila", "leader", "mayda", "nemo", "pg", "sf", "mccain", "lambweston",
}

FAMILY_QUERIES = {
    "french fries",
    "chicken",
    "chicken leg",
    "nuggets",
    "sausage",
    "shawarma",
}


def normalize_catalog_text(text: str) -> str:
    cleaned = str(text or "").lower()
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s*(mm|kg|gm|g)\b", r"\1\2", cleaned)
    cleaned = re.sub(r"[^a-z0-9./\s-]", " ", cleaned)
    cleaned = re.sub(r"\b(?:ka|ki|ke|wala|waala|price|rate|cost|available|stock|ctn|carton|box)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    for source, target in ALIASES.items():
        cleaned = re.sub(rf"\b{re.escape(source)}\b", target, cleaned)

    cleaned = re.sub(r"\bfrench\s+french\s+fries\b", "french fries", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


@lru_cache(maxsize=1)
def _catalog_rows() -> tuple[dict, ...]:
    if not CATALOG_PATH.exists():
        return ()

    rows = []
    with CATALOG_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")
        for row in reader:
            name = (row.get("Name") or "").strip()
            if not name:
                continue
            rows.append({
                "id": row.get("Product ID", ""),
                "name": name,
                "category": row.get("Category", ""),
                "quantity": row.get("Quantity", ""),
                "normalized": normalize_catalog_text(name),
            })
    return tuple(rows)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+|\d+(?:\.\d+)?(?:mm|kg|gm|g)?", text))


def _variant_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?(?:mm|kg|gm|g)\b", text))


def _score(query: str, candidate: str) -> float:
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0

    variants = _variant_tokens(query)
    if variants and not variants.issubset(candidate_tokens):
        return 0.0

    coverage = len(query_tokens & candidate_tokens) / len(query_tokens)
    similarity = SequenceMatcher(None, query, candidate).ratio()
    starts_bonus = 0.12 if candidate.startswith(query) or query.startswith(candidate) else 0.0
    return (coverage * 0.72) + (similarity * 0.28) + starts_bonus


def refine_product_query(query: str) -> str:
    """
    Return a compact, API-friendly product query.

    Generic families stay generic (e.g. "frys" -> "french fries") so the API
    can still prioritize previous-order matches. Variant-only family queries
    stay compact (e.g. "fries 6mm" -> "french fries 6mm").
    """
    normalized = normalize_catalog_text(query)
    if not normalized:
        return str(query or "").strip()

    if normalized in FAMILY_QUERIES:
        return normalized

    if "french fries" in normalized:
        tokens = _tokens(normalized)
        variants = sorted(_variant_tokens(normalized))
        brand_tokens = sorted(tokens & FRIES_BRANDS)
        descriptive_tokens = [
            token for token in ("coated", "uncoated", "crinkle", "spicy")
            if token in tokens
        ]

        # Keep broad fries requests broad enough for backend customer-history
        # matching. Only preserve variants/brands the customer actually typed.
        if not brand_tokens:
            compact = " ".join(["french fries", *variants, *descriptive_tokens]).strip()
            return compact or "french fries"

        return " ".join(["french fries", *brand_tokens, *variants, *descriptive_tokens]).strip()

    rows = _catalog_rows()
    if not rows:
        return normalized

    best_row = None
    best_score = 0.0
    for row in rows:
        score = _score(normalized, row["normalized"])
        if score > best_score:
            best_score = score
            best_row = row

    if best_row and best_score >= 0.68:
        return best_row["name"]

    return normalized
