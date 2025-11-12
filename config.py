"""
Configuration for LLM provider settings.
Supports multiple providers with OpenAI-compatible APIs.
"""

import os
from typing import Optional
from openai import OpenAI, AsyncOpenAI


# Supported providers
# OpenAI is the default (required for APO algorithm)
PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    # Groq (free alternative) - commented but available for testing
    # Note: APO algorithm requires OpenAI models for critique/rewrite steps
    # "groq": {
    #     "base_url": "https://api.groq.com/openai/v1",
    #     "api_key_env": "GROQ_API_KEY",
    #     "default_model": "llama-3.1-8b-instant",
    # },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "default_model": "meta-llama/Llama-3-8b-chat-hf",  # Free tier available
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "google/gemma-2-2b-it:free",  # Free model
    },
    "huggingface": {
        "base_url": "https://api-inference.huggingface.co/v1",
        "api_key_env": "HUGGINGFACE_API_KEY",
        "default_model": "mistralai/Mistral-7B-Instruct-v0.2",  # Free tier available
    },
}


def get_provider() -> str:
    """
    Get the provider name from environment variable, default to openai.
    OpenAI is required for APO algorithm (uses gpt-5-mini for critique/rewrite).
    """
    return os.getenv("LLM_PROVIDER", "openai").lower()


def get_client() -> OpenAI:
    """
    Create an OpenAI-compatible client for the configured provider.
    Defaults to OpenAI (required for APO algorithm).
    """
    provider_name = get_provider()
    
    if provider_name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Supported providers: {', '.join(PROVIDERS.keys())}"
        )
    
    provider = PROVIDERS[provider_name]
    api_key = os.getenv(provider["api_key_env"])
    
    if not api_key:
        raise ValueError(
            f"API key not found for provider '{provider_name}'. "
            f"Please set {provider['api_key_env']} environment variable. "
            f"See README.md for setup instructions."
        )
    
    return OpenAI(
        base_url=provider["base_url"],
        api_key=api_key,
    )


def get_async_client() -> AsyncOpenAI:
    """
    Create an async OpenAI-compatible client for the configured provider.
    """
    provider_name = get_provider()
    
    if provider_name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Supported providers: {', '.join(PROVIDERS.keys())}"
        )
    
    provider = PROVIDERS[provider_name]
    api_key = os.getenv(provider["api_key_env"])
    
    if not api_key:
        raise ValueError(
            f"API key not found for provider '{provider_name}'. "
            f"Please set {provider['api_key_env']} environment variable."
        )
    
    return AsyncOpenAI(
        base_url=provider["base_url"],
        api_key=api_key,
    )


def get_default_model() -> str:
    """Get the default model for the configured provider."""
    provider_name = get_provider()
    return PROVIDERS[provider_name]["default_model"]

