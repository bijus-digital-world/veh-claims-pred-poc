"""
Intent-based fallback handlers for non-data queries.
"""

from __future__ import annotations

from chat.handlers import QueryHandler, QueryContext
from utils.logger import chat_logger as logger


GENERIC_RESPONSES = {
    "small_talk": (
        "<p>I'm the Nissan Telematics Analytics Assistant, "
        "so I focus on vehicle data questions rather than personal topics. "
        "Feel free to ask me about failures, warranty risk, suppliers, or other telematics analytics.</p>"
    ),
    "off_domain": (
        "<p>I’m specialized for Nissan telematics analytics. "
        "Please ask about vehicle models, failures, suppliers, or other data-driven topics, "
        "and I’ll dig into the dataset for you.</p>"
    ),
    "safety": (
        "<p>I can’t help with that request. "
        "Let me know if you need insights about vehicle reliability or warranty risk instead.</p>"
    ),
    "empty": "<p>Please enter a question about the telematics dataset.</p>",
}


class GenericIntentHandler(QueryHandler):
    """Respond generically when the query is outside the telematics domain."""

    def can_handle(self, context: QueryContext) -> bool:
        return context.intent in {"small_talk", "off_domain", "safety"}

    def handle(self, context: QueryContext) -> str:
        logger.debug(f"GenericIntentHandler responding for intent={context.intent}")
        message = GENERIC_RESPONSES.get(
            context.intent or "off_domain",
            GENERIC_RESPONSES["off_domain"],
        )
        return message

