"""
Singleton Bedrock client for efficient LLM API calls.

Reuses a single boto3 client across all handlers to avoid recreation overhead.
"""

import boto3
from botocore.config import Config
from typing import Optional
from config import config
from utils.logger import chat_logger as logger

# Singleton Bedrock client
_bedrock_client: Optional[boto3.client] = None

# Keep boto3's retry footprint small. Bedrock throttling on a quota-exhausted
# account does not recover within seconds, so retrying just slows chat to a
# crawl while we wait to fail. With max_attempts=2 (i.e. 1 retry), a throttled
# call returns in ~200 ms instead of ~10 s, and the app's own fallback (pattern
# SQL / simple formatting) kicks in immediately. When Bedrock is healthy, calls
# succeed on the first try so retry budget is unused.
_BEDROCK_RETRY_CONFIG = Config(retries={"max_attempts": 1, "mode": "standard"})


def get_bedrock_client() -> boto3.client:
    """Get or create singleton Bedrock client with conservative retry config."""
    global _bedrock_client

    if _bedrock_client is None:
        try:
            _bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=config.aws.region,
                config=_BEDROCK_RETRY_CONFIG,
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


# ---- Model-family-agnostic request/response helpers ----
# Different Bedrock model families expect different invoke_model body shapes.
# These helpers centralize the schema so callers don't repeat it.

def _model_family(model_id: str) -> str:
    """Return 'nova' | 'titan' | 'claude' based on the model_id."""
    m = (model_id or "").lower()
    if "nova" in m:
        return "nova"
    if "titan" in m:
        return "titan"
    return "claude"


def build_text_invoke_body(model_id: str, prompt: str,
                           max_tokens: int = 500, temperature: float = 0.1) -> dict:
    """Build the invoke_model body dict for the given model_id and prompt."""
    family = _model_family(model_id)
    if family == "nova":
        return {
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
    if family == "titan":
        return {
            "inputText": prompt,
            "textGenerationConfig": {"maxTokenCount": max_tokens, "temperature": temperature},
        }
    # default: Anthropic Claude
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }


def parse_text_invoke_response(model_id: str, response_body: dict) -> str:
    """Extract the assistant text from an invoke_model response body."""
    family = _model_family(model_id)
    if family == "nova":
        try:
            return response_body["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return ""
    if family == "titan":
        results = response_body.get("results") or []
        if isinstance(results, list) and results:
            return results[0].get("outputText") or results[0].get("text") or ""
        return response_body.get("outputText", "")
    # default: Claude
    try:
        return response_body["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""

