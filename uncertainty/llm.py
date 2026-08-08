"""Building a chat model for one provider at one effort tier.

classifier.llm.build_model reads a fixed REASONING_EFFORT from classifier.config. The UQ
runs need a different effort per tier and a different model id, so both are overridden
here rather than by mutating the classifier's configuration.

This lives in the package rather than in uq_main.py so that probe.py can import it
without a root-level script being imported back into the package.

As in classifier/llm.py, the model is returned rather than used, so callers can be tested
with a fake (REFERENCE.md REF3c).
"""

from langchain_core.language_models import BaseChatModel

from classifier.config import BASE_URLS, Provider, Settings
from uncertainty.config import UQ_MODEL_IDS, Tier, effort_for


def build_tier_model(provider: Provider, tier: Tier, settings: Settings) -> BaseChatModel:
    """The chat model for one provider at one canonical effort tier."""
    model_id = UQ_MODEL_IDS[provider]
    effort = effort_for(provider, tier)
    api_key = settings.key_for(provider)

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        # 'none' means no reasoning: omit the parameter rather than send a bad value.
        options = {} if effort == "none" else {"reasoning_effort": effort}
        return ChatMistralAI(model=model_id, api_key=api_key, **options)

    from langchain_openai import ChatOpenAI

    kwargs: dict[str, object] = {
        "model": model_id,
        "api_key": api_key,
        "reasoning_effort": effort,
    }
    if provider == "kimi":
        kwargs["base_url"] = BASE_URLS["kimi"]
    return ChatOpenAI(**kwargs)
