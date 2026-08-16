"""Building a chat model for each provider.

Every provider returns a BaseChatModel, which is already the abstraction the rest
of the code needs - so there is no wrapper class here on purpose (REFERENCE.md
REF1: the best refactoring is removing code).

The model is built here and passed into classify_case() as an argument rather
than being constructed inside it, which is what lets the tests inject a fake
model and run with no API calls (REF3c).
"""

from langchain_core.language_models import BaseChatModel

from classifier.config import Provider, Settings, BASE_URLS, REASONING_EFFORT


def build_model(provider: Provider, settings: Settings) -> BaseChatModel:
    """ Create the chat model for one provider.
    
    """
    model_id = settings.model_for(provider)
    api_key = settings.key_for(provider)

    #options: dict[str, object] = {"reasoning_effort": REASONING_EFFORT[provider]}
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        #return ChatOpenAI(model= model_id, api_key= api_key, **options)
        return ChatOpenAI(model= model_id, api_key= api_key, reasoning_effort= REASONING_EFFORT[provider])
    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model = model_id, api_key= api_key, reasoning_effort= REASONING_EFFORT[provider])
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model= model_id, api_key= api_key, temperature= 0, reasoning_format= "parsed", reasoning_effort= REASONING_EFFORT[provider])
    
    if provider == "kimi":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model= model_id, 
            api_key= api_key,
            base_url= BASE_URLS["kimi"],
            reasoning_effort= REASONING_EFFORT[provider],
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model = model_id,
            temperature= 0,
        )
    raise ValueError(f"Unknown provider: {provider!r}") 

