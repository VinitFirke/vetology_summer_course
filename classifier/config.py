"""Configuration: where files live, which models to call, and API keys.

Keys are held as SecretStr so they cannot be printed by accident - a plain str
would show up in any traceback or debug print that touches Settings.
"""

from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, SecretStr

Provider = Literal["openai", "mistral", "groq", "kimi", "ollama_gemma", "ollama_nemotron"]

PROVIDERS: tuple[Provider, ...] = ("openai", "mistral", "groq", "kimi", "ollama_gemma", "ollama_nemotron")

# Model IDs, verified available on these accounts. Swap freely.
DEFAULT_MODEL_IDS: dict[Provider, str] = {
    "openai": "gpt-5.5",
    "mistral": "mistral-medium-latest",
    "groq": "openai/gpt-oss-120b",
    "kimi": "kimi-k3",
    "ollama_gemma": "gemma4",
    "ollama_nemotron": "nemotron-3-super:cloud"
}   
#"groq": "llama-3.3-70b-versatile",

REASONING_EFFORT: dict[Provider, str] = {
    "openai": "medium",
    "mistral": "medium",
    "groq": "medium",
    "kimi": "high",
}
BASE_URLS: dict[Provider, str] = {
    "kimi": "https://api.moonshot.ai/v1",
}

# Project layout.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset_LLM_classification"
INPUT_CSV = DATA_DIR / "feline_thorax_manual_final_50_appended_for_LLM_classification(in).csv"
PROMPT_FILE = PROJECT_ROOT / "prompts" / "classification_prompt.json"
LOG_FILE = PROJECT_ROOT / "logs" / "token_usage.md"

# Evaluation: the manually scored ground truth, and where the matrix is written.
GOLD_DIR = PROJECT_ROOT / "dataset_gold_standard"
GOLD_CSV = GOLD_DIR / "feline_thorax_manual_final_50_appended(in).csv"
# Written alongside the other outputs, so the gold standard folder stays read-only.
CONFUSION_MATRIX_XLSX = DATA_DIR / "confusion_matrix_feline_thorax.xlsx"

# The gold file predates a column rename in the prediction files. Kept here, in one
# visible place, rather than hidden inside the comparison logic.
GOLD_TO_PREDICTION: dict[str, str] = {
    "Fe_Alveolar": "Alveolar_interstitial_pattern",
}

ENV_VAR_NAMES: dict[Provider, str] = {
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "kimi":"KIMI_API_KEY",
}


class Settings(BaseModel):
    """API keys and model IDs, validated once at startup."""

    api_keys: dict[str, SecretStr]
    model_ids: dict[str, str]

    def key_for(self, provider: Provider) -> str:
        """Return the API key for a provider, or explain clearly what is missing."""
        if provider == "ollama_gemma" or provider == "ollama_nemotron" :
            return ""
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

    Reads the file directly rather than mutating os.environ, so tests can point at
    a different file without leaking state between them.
    """
    env_path = env_file or (PROJECT_ROOT / ".env")
    values = dotenv_values(env_path)

    api_keys: dict[str, SecretStr] = {}
    for provider in PROVIDERS:
        if provider == "ollama_gemma" or provider == "ollama_nemotron":
            api_keys[provider] = SecretStr("")
            continue           
        raw = values.get(ENV_VAR_NAMES[provider]) or ""
        api_keys[provider] = SecretStr(raw.strip())

    return Settings(api_keys=api_keys, model_ids=dict(DEFAULT_MODEL_IDS))


def output_csv_path(provider: Provider) -> Path:
    return DATA_DIR / f"feline_thorax_labeled_{provider}.csv"


def reasoning_json_path(provider: Provider) -> Path:
    return DATA_DIR / f"reasoning_{provider}.json"
