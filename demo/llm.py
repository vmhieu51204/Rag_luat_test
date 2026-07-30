"""LLM configuration and helper methods for the Streamlit demo."""

import os
from rag.llm.providers import LLMProvider, default_model_for_provider

def get_available_providers() -> list[str]:
    """Returns a list of supported LLM provider names."""
    return [p.value for p in LLMProvider]

def get_default_model(provider_name: str) -> str:
    """Returns the default model name for a given provider."""
    try:
        provider = LLMProvider(provider_name)
        return default_model_for_provider(provider)
    except ValueError:
        return ""

def configure_api_keys(api_key: str, provider_name: str) -> None:
    """Sets the API key in environment variables based on the provider."""
    if not api_key:
        return
    
    if provider_name == "aistudio":
        os.environ["GOOGLE_API_KEY"] = api_key
    elif provider_name == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = api_key
    elif provider_name == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
