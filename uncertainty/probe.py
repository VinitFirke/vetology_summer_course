"""Does this provider actually return token logprobs?

TLP needs the API to hand back the probability it assigned to each emitted token.
Reasoning models often decline, and the field is omitted silently rather than raising -
so this asks the question with three real calls instead of an assumption.

    python -m uncertainty.probe

Costs about $0.05 in total: one case per provider. Build TLP (plan.md Task 14) only for
providers that answer YES.
"""

from typing import Any

from classifier import config as classifier_config
from classifier.config import Provider, Settings, load_settings
from classifier.csv_io import read_cases
from classifier.prompt import load_prompt, render_messages
from uncertainty import config as uq_config
from uncertainty.llm import build_tier_model
from uncertainty.schemas import CaseLabels


def describe_logprobs(raw: Any) -> tuple[bool, str]:
    """Inspect a raw AIMessage for usable token logprobs. Returns (supported, detail).

    Pure, so the awkward part - working out where each provider hides the field, and
    what "present but useless" looks like - is testable without spending anything.

    LangChain surfaces OpenAI-style logprobs at response_metadata["logprobs"], usually a
    dict with a "content" list of {token, logprob} pairs. Some OpenAI-compatible
    providers return the list directly, and some return the key with a null value.
    """
    metadata = getattr(raw, "response_metadata", None) or {}
    logprobs = metadata.get("logprobs")

    if logprobs is None:
        return False, "response carried no logprobs field"

    content = logprobs.get("content") if isinstance(logprobs, dict) else logprobs
    if not content:
        return False, "logprobs field present but empty"

    first = content[0]
    if not isinstance(first, dict) or "logprob" not in first:
        return False, f"logprobs present but unrecognised shape: {type(first).__name__}"

    return True, f"{len(content)} token logprobs returned"


def probe_logprobs(provider: Provider, tier: str, settings: Settings) -> tuple[bool, str]:
    """Send one case with logprobs requested. Returns (supported, detail).

    A rejected parameter is a valid answer, not a crash, so any exception is caught and
    reported as an unsupported result.
    """
    case = read_cases(classifier_config.INPUT_CSV)[0]
    prompt = load_prompt(classifier_config.PROMPT_FILE)

    try:
        model = build_tier_model(provider, tier, settings).bind(logprobs=True)
        structured = model.with_structured_output(CaseLabels, include_raw=True)
        response = structured.invoke(render_messages(prompt, case))
    except Exception as error:  # noqa: BLE001 - a rejected parameter is a valid answer
        return False, f"call failed: {type(error).__name__}: {error}"

    return describe_logprobs(response.get("raw"))


def main() -> None:
    settings = load_settings()
    print("Probing for token logprobs - one call per provider, ~$0.05 total.\n")

    results: dict[str, bool] = {}
    for provider in uq_config.UQ_PROVIDERS:
        supported, detail = probe_logprobs(provider, "low", settings)
        results[provider] = supported
        print(f"  {provider:<8} {'YES' if supported else 'NO ':<4} {detail}")

    yes = [p for p, ok in results.items() if ok]
    print()
    if yes:
        print(f"Build TLP for: {', '.join(yes)}")
        print("Set LOGPROB_PROVIDERS in uncertainty/config.py, then do plan.md Task 14.")
    else:
        print("No provider returned logprobs. Skip Task 14; CE and SC are the final set.")
    print("Record the outcome under D5 in catalog.md either way.")


if __name__ == "__main__":
    main()
