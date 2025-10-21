"""
chat package

Modular chat handler system for the Vehicle Insight Companion.
Replaces the monolithic generate_reply() god function with focused, testable handlers.
"""

from chat.handlers import QueryRouter, QueryContext

__all__ = ['QueryRouter', 'QueryContext']

