"""
Singleton Bedrock client for efficient LLM API calls.

Reuses a single boto3 client across all handlers to avoid recreation overhead.
"""

import boto3
from botocore.exceptions import ClientError
from typing import Optional
from config import config
from utils.logger import chat_logger as logger

# Singleton Bedrock client
_bedrock_client: Optional[boto3.client] = None


def get_bedrock_client() -> boto3.client:
    """
    Get or create singleton Bedrock client.
    
    Returns:
        Reusable boto3 Bedrock client
    """
    global _bedrock_client
    
    if _bedrock_client is None:
        try:
            _bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=config.aws.region
            )
            logger.debug(f"Created singleton Bedrock client for region: {config.aws.region}")
        except Exception as e:
            logger.error(f"Failed to create Bedrock client: {e}")
            raise
    
    return _bedrock_client


def reset_bedrock_client():
    """Reset the singleton client (useful for testing or reconnection)."""
    global _bedrock_client
    _bedrock_client = None

