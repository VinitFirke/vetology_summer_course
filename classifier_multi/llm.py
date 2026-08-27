"""Building a chat model for each provider.

Every provider returns a BaseChatModel, which is already the abstraction the rest
of the code needs - so there is no wrapper class here on purpose (REFERENCE.md
REF1: the best refactoring is removing code).

The model is built here and passed into classify_case() as an argument rather
than being constructed inside it, which is what lets the tests inject a fake
model and run with no API calls (REF3c).
"""

from langchain_core.language_models import BaseChatModel

from classifier_multi.config import Provider, Settings, BASE_URLS

# How long one request may take before it is abandoned and retried. See build_model.
REQUEST_TIMEOUT_SECONDS = 180


def build_model(provider: Provider, settings: Settings) -> BaseChatModel:
    """ Create the chat model for one provider.
    
    """
    model_id = settings.model_for(provider)
    api_key = settings.key_for(provider)

    # Ollama Cloud is OpenAI-wire-compatible, so all four providers are the same
    # client pointed at the same endpoint - only the model ID differs. ChatOllama is
    # not used any more: it talks to the local daemon, not the hosted models.
    if provider in BASE_URLS:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model= model_id,
            api_key= api_key,
            base_url= BASE_URLS[provider],
            temperature= 0,
            # Without a timeout a throttled request just hangs. During one long sweep
            # the endpoint slowed to 103s for a five-token call, and calls that would
            # normally take 5s sat open instead of failing and being retried. Capped
            # well above the slowest healthy call measured (89s for a 19-finding
            # thorax case on qwen) so a slow model is not mistaken for a stuck one.
            timeout= REQUEST_TIMEOUT_SECONDS,
            # The retry policy lives in classify.invoke_structured, which knows which
            # failures are worth retrying. Leaving the client's own retries on would
            # multiply the two and hide the real error behind the wrong exception.
            max_retries= 0,
        )

    raise ValueError(f"Unknown provider: {provider!r}")

