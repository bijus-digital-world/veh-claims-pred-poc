"""
Lightweight intent classifier for chat queries.

The classifier identifies whether a query is a telematics data request or
should be handled generically (small talk / off-domain / safety).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

IntentLabel = Literal["empty", "data_request", "small_talk", "off_domain", "safety"]


DOMAIN_KEYWORDS = [
    "failure",
    "failures",
    "failing",
    "failed",
    "claim",
    "claims",
    "repair",
    "repairs",
    "recall",
    "recalls",
    "telematics",
    "vehicle",
    "vehicles",
    "fleet",
    "battery",
    "batteries",
    "voltage",
    "soc",
    "temperature",
    "model",
    "models",
    "part",
    "parts",
    "pfp",
    "primary_failed_part",
    "primary failed part",
    "component",
    "components",
    "sentra",
    "leaf",
    "ariya",
    "altima",
    "rogue",
    "pathfinder",
    "frontier",
    "titan",
    "mileage",
    "age",
    "supplier",
    "suppliers",
    "dealer",
    "dealers",
    "service center",
    "service centers",
    "nearest",
    "dtc",
    "fault",
    "faults",
    "warning",
    "anomaly",
    "anomalies",
    "risk",
    "warranty",
    "histogram",
    "trend",
    "trends",
    "rate",
    "rates",
    "count",
    "ranking",
    "rank",
    "top",
    "worst",
    "best",
    "most",
    "least",
    "records",
    "dataset",
    "data",
    # VIN and location keywords
    "vin",
    "vins",
    "location",
    "locations",
    "coordinates",
    "latitude",
    "longitude",
    "lat",
    "lon",
    "city",
    "where",
    # Schema/metadata keywords
    "column",
    "columns",
    "field",
    "fields",
    "schema",
    "feature",
    "features",
    "header",
    "headers",
    "available",
    "structure",
]

SMALL_TALK_PATTERNS = [
    r"\bwho are you\b",
    r"\bwhat('?s| is)? your name\b",
    r"\bhow old are you\b",
    r"\bwhere do you live\b",
    r"\bwhere do you work\b",
    r"\b(girl|boy) ?friend\b",
    r"\bwife\b",
    r"\bhusband\b",
    r"\bspouse\b",
    r"\bmarried\b",
    r"\bwhich school\b",
    r"\bcollege\b",
    r"\bwho made you\b",
    r"\bare you (male|female|a boy|a girl)\b",
]

SAFETY_PATTERNS = [
    r"\bpassword\b",
    r"\bcredit card\b",
    r"\bssn\b",
    r"\bsocial security\b",
]


@dataclass
class IntentResult:
    label: IntentLabel
    reason: str = ""


def classify_intent(query: str) -> IntentResult:
    """Classify the query into a coarse intent label."""
    if not query or not query.strip():
        return IntentResult(label="empty", reason="blank query")

    q = query.lower().strip()

    for pattern in SAFETY_PATTERNS:
        if re.search(pattern, q):
            return IntentResult(label="safety", reason="safety pattern detected")

    for pattern in SMALL_TALK_PATTERNS:
        if re.search(pattern, q):
            return IntentResult(label="small_talk", reason=f"matched small-talk pattern: {pattern}")

    # Check for schema/metadata queries (e.g., "what columns are available", "show schema")
    schema_patterns = [
        r'\b(what|which|show|list|display)\s+(columns?|fields?|schema|features?|headers?)\b',
        r'\b(columns?|fields?|schema|features?|headers?)\s+(available|in|are|do|have)\b',
        r'\b(available|in|are|do|have)\s+(columns?|fields?|schema|features?|headers?)\b',
    ]
    for pattern in schema_patterns:
        if re.search(pattern, q):
            return IntentResult(label="data_request", reason="schema/metadata query detected")
    
    domain_hits = sum(1 for keyword in DOMAIN_KEYWORDS if keyword in q)
    question_word = bool(re.search(r"\b(what|how|show|find|list|give|count|average|why|when|where|which|is|are|there|does|do|can|could|will|would)\b", q))
    
    # Check for "is there" / "are there" queries with domain context
    existence_query = bool(re.search(r'\b(is|are)\s+(there|a|an|any)\b', q))
    
    # Check for ranking queries with domain context (e.g., "top 5 failing parts")
    ranking_with_domain = bool(re.search(r'\b(top|worst|best|most|least|ranking|rank)\s+\d+.*?\b(part|component|model|vehicle|failure|claim|repair)\b', q))

    if domain_hits >= 1 and question_word:
        return IntentResult(label="data_request", reason="contains domain keywords with question word")

    if domain_hits >= 1 and existence_query:
        return IntentResult(label="data_request", reason="existence query with domain keywords")

    if domain_hits >= 2:
        return IntentResult(label="data_request", reason="multiple domain keywords detected")
    
    if ranking_with_domain:
        return IntentResult(label="data_request", reason="ranking query with domain context")

    return IntentResult(label="off_domain", reason="no domain context detected")

