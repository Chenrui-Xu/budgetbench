"""Shared Gemini client + retry helper for all agents."""
from __future__ import annotations

import os
import time

from google import genai
from google.genai.errors import ServerError


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "ai-agent-for-engineers"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


def generate_with_retry(
    model: str,
    contents: str,
    max_retries: int = 5,
    base_delay: float = 5.0,
) -> str:
    """Call generate_content with exponential backoff on 503/429."""
    client = get_client()
    delay = base_delay
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
            return response.text or ""
        except ServerError as e:
            if e.code in (429, 503) and attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return ""
