"""
Context utilities for conversation memory integration.

Provides generic utilities for extracting context from conversation history
that can be used by any handler.
"""

import re
import json
from typing import Dict, Optional, List
from chat.handlers import QueryContext
from chat.bedrock_client import get_bedrock_client
from utils.logger import chat_logger as logger
from config import config


def extract_context_filters_from_memory(
    context: QueryContext, 
    recent_exchanges: List,
    filter_types: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Use Bedrock LLM to intelligently extract context filters from conversation memory.
    
    This is a generic utility that can extract any type of context information
    from previous exchanges, not just failure-related filters.
    
    Args:
        context: The current query context
        recent_exchanges: List of recent conversation exchanges
        filter_types: Optional list of filter types to extract. If None, extracts
                     common filters: part_family, age_bucket, mileage_bucket
    
    Returns:
        Dictionary with extracted filters (e.g., {'part_family': 'Battery', 'age_bucket': '1-3yr'})
    """
    if not recent_exchanges:
        return {}
    
    if filter_types is None:
        filter_types = ['part_family', 'age_bucket', 'mileage_bucket']
    
    try:
        from botocore.exceptions import ClientError
        
        # Build context from recent exchanges
        context_text = ""
        for exchange in reversed(recent_exchanges[:3]):  # Use up to 3 most recent
            context_text += f"Previous Query: {exchange.query}\n"
            # Remove HTML tags from response for cleaner context
            clean_response = re.sub(r'<[^>]+>', '', exchange.response)
            context_text += f"Previous Response: {clean_response[:500]}\n\n"
        
        # Build prompt based on filter types
        filter_definitions = {
            'part_family': 'Battery|Brakes|Transmission|Engine|Electrical|Lighting|HVAC|Safety|Steering|Tires',
            'age_bucket': '1-3yr|3-5yr|5-7yr|7+yr',
            'mileage_bucket': '0-30k|30-60k|60-90k|90+k',
        }
        
        json_structure = {}
        for filter_type in filter_types:
            if filter_type in filter_definitions:
                json_structure[filter_type] = f"{filter_definitions[filter_type]} or null"
            else:
                json_structure[filter_type] = "string or null"
        
        # Build prompt for LLM to extract filters
        prompt = f"""You are analyzing a conversation about vehicle data. The user is asking about something that refers to information mentioned in previous exchanges.

Previous conversation context:
{context_text}

Current query: {context.query}

Based on the previous conversation, extract the specific criteria that the current query refers to. Return a JSON object with the following structure:
{json.dumps(json_structure, indent=2)}

Only include fields that are explicitly mentioned or clearly implied in the previous conversation. If a field is not mentioned, set it to null.
Return ONLY the JSON object, no additional text.

Example 1:
Previous: "At what vehicle age do battery failures occur most frequently?"
Response: {{"part_family": "Battery", "age_bucket": null, "mileage_bucket": null}}

Example 2:
Previous: "Battery failures occur in vehicles aged 1-3 years"
Response: {{"part_family": "Battery", "age_bucket": "1-3yr", "mileage_bucket": null}}

Example 3:
Previous: "What is the failure rate for transmission issues in vehicles with 30-60k mileage?"
Response: {{"part_family": "Transmission", "age_bucket": null, "mileage_bucket": "30-60k"}}

JSON:"""
        
        bedrock = get_bedrock_client()  # Use singleton client
        model_id = config.model.bedrock_model_id
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "temperature": 0.1,  # Low temperature for consistent extraction
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        }
        
        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        
        response_body = json.loads(response['body'].read())
        llm_response = response_body['content'][0]['text'].strip()
        
        # Extract JSON from response (handle cases where LLM adds extra text)
        json_match = re.search(r'\{[^}]+\}', llm_response, re.DOTALL)
        if json_match:
            filters = json.loads(json_match.group(0))
            # Remove null values
            filters = {k: v for k, v in filters.items() if v is not None}
            logger.info(f"LLM extracted context filters: {filters}")
            return filters
        else:
            logger.warning(f"Could not parse JSON from LLM response: {llm_response}")
            return {}
        
    except ClientError as e:
        logger.error(f"Bedrock API error for context extraction: {e}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in context extraction: {e}")
        return {}
    except Exception as e:
        logger.error(f"Context extraction failed: {e}", exc_info=True)
        return {}


def extract_referenced_entity_from_memory(
    context: QueryContext,
    recent_exchanges: List,
    entity_type: str = "VIN"
) -> Optional[str]:
    """
    Extract a referenced entity (like VIN, model, part) from conversation memory.
    
    Args:
        context: The current query context
        recent_exchanges: List of recent conversation exchanges
        entity_type: Type of entity to extract (e.g., "VIN", "model", "part")
    
    Returns:
        The extracted entity value or None if not found
    """
    if not recent_exchanges:
        return None
    
    # For VIN, use regex pattern matching (more reliable than LLM for structured data)
    if entity_type.upper() == "VIN":
        for exchange in reversed(recent_exchanges):
            vin_match = re.search(r'1N4[A-Z0-9]{8,17}', exchange.query)
            if vin_match:
                return vin_match.group(0)
            vin_match = re.search(r'1N4[A-Z0-9]{8,17}', exchange.response)
            if vin_match:
                return vin_match.group(0)
        return None
    
    # For other entity types, could use LLM if needed
    # For now, return None for non-VIN entities
    return None


def is_context_reference(query: str, reference_patterns: Optional[List[str]] = None) -> bool:
    """
    Check if a query contains references to previous context.
    
    Args:
        query: The query text to check
        reference_patterns: Optional list of regex patterns. If None, uses default patterns.
    
    Returns:
        True if the query contains context references
    """
    query_lower = query.lower()
    
    if reference_patterns is None:
        # Default patterns for common references
        reference_patterns = [
            r'\b(these|those|the)\s+(failures?|vehicles?|models?|parts?)\b',
            r'\b(this|that|the)\s+(vin|vehicle|model|part)\b',
            r'\b(failures?|vehicles?|models?|parts?)\s+(mentioned|above|before|earlier)\b',
        ]
    
    return any(re.search(pattern, query_lower) for pattern in reference_patterns)

