"""Configuration: where files live, which models to call, and API keys.

Same shape as the single-category classifier's config, with two differences:

  * every path is a function of a Category, so one run's outputs cannot land on top
    of another study type's;
  * no reasoning_effort is sent. The four providers run on their own defaults, which
    is what this comparison is meant to measure;
  * nothing here knows the manually scored spreadsheets exist. Their location, their
    filenames and the vocabulary their scorers used belong to evaluate.py, so a
    classification run cannot reach the answers even by accident.

Keys are held as SecretStr so they cannot be printed by accident - a plain str would
show up in any traceback or debug print that touches Settings.
"""

from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, SecretStr

from classifier_multi.categories import Category

# Four models served from Ollama Cloud. They replace the local ollama_* providers,
# which ran the same weight classes on this machine's GPU.
Provider = Literal["cloud_gemma", "cloud_qwen", "cloud_kimi", "cloud_nemotron"]

PROVIDERS: tuple[Provider, ...] = ("cloud_gemma", "cloud_qwen", "cloud_kimi", "cloud_nemotron")
# Model IDs, each verified callable on this account.
#
# openai: the request was "GPT 5.5 Terra". No gpt-5.5-terra exists - the Terra variant
# is published at 5.6 - so the Terra line is what is pinned here. Change this one line
# to "gpt-5.5" if the plain 5.5 model was what was wanted.
# Model IDs exactly as Ollama Cloud publishes them - each one verified present in
# GET https://ollama.com/v1/models on this account. The tag is part of the ID; an
# untagged "gemma4" is not a cloud model and will 404.
DEFAULT_MODEL_IDS: dict[Provider, str] = {
    "cloud_gemma": "gemma4:31b",
    "cloud_qwen": "qwen3.5:397b",
    "cloud_kimi": "kimi-k2.6",
    "cloud_nemotron": "nemotron-3-super",
}

# Ollama Cloud speaks the OpenAI wire format, so every provider shares one endpoint
# and one key - unlike the local daemon, which needed neither.
OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"

BASE_URLS: dict[Provider, str] = {p: OLLAMA_CLOUD_BASE_URL for p in PROVIDERS}

ENV_VAR_NAMES: dict[Provider, str] = {p: "OLLAMA_API_KEY" for p in PROVIDERS}

# Project layout. The dataset directories are shared with the single-category
# classifier; the code is not.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset_LLM_classification"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


# The two prompt styles. The variant is a directory rather than a filename suffix, so
# a run of one style can never land on top of the other's results.
Variant = Literal["zeroshot", "fewshot"]

VARIANTS: tuple[Variant, ...] = ("zeroshot", "fewshot")


def short_name(provider: Provider) -> str:
    """`cloud_gemma` -> `gemma`. Filenames only; provider keys keep the prefix."""
    return provider.removeprefix("cloud_")
LOG_FILE = PROJECT_ROOT / "logs" / "token_usage_multi.md"


class Settings(BaseModel):
    """API keys and model IDs, validated once at startup."""

    api_keys: dict[str, SecretStr]
    model_ids: dict[str, str]

    def key_for(self, provider: Provider) -> str:
        """Return the API key for a provider, or explain clearly what is missing.

        Every provider is now a hosted one, so a missing key is always an error -
        the local-Ollama exemption that used to live here is gone.
        """
        key = self.api_keys.get(provider)
        if key is None or not key.get_secret_value().strip():
            raise ValueError(
                f"No API key for '{provider}'. "
                f"Add {ENV_VAR_NAMES[provider]}=... to {PROJECT_ROOT / '.env'}"
            )
        return key.get_secret_value()

    def model_for(self, provider: Provider) -> str:
        return self.model_ids[provider]


def load_settings(env_file: Path | None = None) -> Settings:
    """Read .env and build Settings.

    Reads the file directly rather than mutating os.environ, so tests can point at a
    different file without leaking state between them.
    """
    env_path = env_file or (PROJECT_ROOT / ".env")
    values = dotenv_values(env_path)

    api_keys: dict[str, SecretStr] = {}
    for provider in PROVIDERS:
        raw = values.get(ENV_VAR_NAMES[provider]) or ""
        api_keys[provider] = SecretStr(raw.strip())

    return Settings(api_keys=api_keys, model_ids=dict(DEFAULT_MODEL_IDS))


def input_csv_path(category: Category) -> Path:
    return DATA_DIR / category.input_csv


def prompt_path(category: Category, variant: Variant = "zeroshot") -> Path:
    """The prompt file for a category and style."""
    suffix = None if variant == "zeroshot" else variant
    return PROMPT_DIR / category.prompt_filename(suffix)


def variant_dir(variant: Variant) -> Path:
    """One directory per prompt style, under the data directory."""
    return DATA_DIR / variant


def predictions_path(category: Category, provider: Provider, variant: Variant) -> Path:
    """Where one run's labelled CSV goes. Never the input file."""
    return variant_dir(variant) / (
        f"{category.name}_classified_{short_name(provider)}.csv"
    )


def reasoning_path(category: Category, provider: Provider, variant: Variant) -> Path:
    """The per-finding evidence and reasoning, beside its predictions."""
    return variant_dir(variant) / (
        f"{category.name}_reasoning_{short_name(provider)}.json"
    )


