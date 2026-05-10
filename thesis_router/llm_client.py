"""LLM provider clients, configured from environment variables.

Set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in a `.env` file (see `.env.example`)
or export them in your shell before running any phase that calls an LLM.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ConfigurationError(RuntimeError):
    """Raised when a required API key or setting is missing."""


def get_openai_client():
    """Return an OpenAI client. Reads `OPENAI_API_KEY` from the environment.

    `OPENAI_BASE_URL` is optional; leave unset to use the default api.openai.com
    endpoint. Override only if you are routing through a proxy or a compatible
    provider (e.g. Azure OpenAI).
    """
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ConfigurationError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key, "
            "or export OPENAI_API_KEY in your shell."
        )
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    return OpenAI(api_key=key, base_url=base_url)


def get_anthropic_client():
    """Return an Anthropic client. Reads `ANTHROPIC_API_KEY` from the environment."""
    from anthropic import Anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ConfigurationError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill in your key, "
            "or export ANTHROPIC_API_KEY in your shell."
        )
    return Anthropic(api_key=key)
