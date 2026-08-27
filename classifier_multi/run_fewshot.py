"""Run every few-shot classification, four models at once, with resume.

`main.py classify` does one category with one provider and sends one request at a
time. That is the right shape for a single run and the wrong shape for the full
sweep: three categories times four models is twelve runs, and done serially the
sweep is bounded by the sum of every call it makes.

The pool sizes here are measured, not guessed. On this account:

    gemma x1                        11.9s
    gemma x3                        all three finished in 7.1s
    gemma x6                        median 6.8s, max 13.2s   <- a 4th call queues
    gemma x3 + nemotron x3          all six started at once, 20.5s

So a model serves three concurrent requests and a fourth waits, while two models
run three-wide side by side without contending. That gives one pool of three per
model, four pools at once, twelve requests in flight.

Each provider walks its own categories rather than the sweep advancing category by
category, because a category-at-a-time sweep would leave gemma idle for hours
waiting on kimi. Total time is then the slowest single model, not the slowest model
summed over categories.

Every finished case is appended to a JSONL checkpoint before the run moves on, so an
interrupted sweep restarts from the case after the last one written rather than from
the beginning. This module stays on the classification side of the boundary the
boundary test enforces: it never reads the manually scored answers.
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from classifier_multi import config
from classifier_multi.categories import CATEGORY_NAMES, Category, get_category
from classifier_multi.classify import ProviderUnavailable, Usage, classify_case
from classifier_multi.csv_io import build_label_row, read_cases, write_labeled_csv
from classifier_multi.llm import build_model
from classifier_multi.prompt import load_prompt

# This runner exists to sweep the few-shot prompts; the zero-shot baseline is what
# `main.py classify` is for. Fixed rather than an argument so a sweep cannot land on
# the baseline's files.
VARIANT: config.Variant = "fewshot"

# Two concurrent requests per model, not the three a short burst supports.
#
# The burst measurements above held for about ninety minutes of a real sweep and then
# stopped holding: latency climbed until a five-token request took 103s, and the sweep
# managed 98 cases in the following eight hours instead of the ~500 it should have.
# Stopping the sweep put latency back to 0.7s within a minute. A burst test says what
# the endpoint will absorb, not what it will keep absorbing, so the sustained figure
# is deliberately below the burst ceiling.
PER_MODEL_CONCURRENCY = 2

# Attempts per case. classify.invoke_structured defaults to 10, which at capped
# backoff is over six minutes spent on a case that is not going to succeed; across a
# throttled sweep that is most of the wall clock. Five still clears the occasional
# unparseable-JSON reply, which is what the retries are actually for.
MAX_ATTEMPTS = 5

# Stop the sweep if fewer than STALL_MIN_COMPLETIONS cases finish in any
# STALL_WINDOW_MINUTES. Healthy throughput here is ~7 cases/minute across four models
# and throttled is ~0.06, so a floor of 5 per 20 minutes is nowhere near either a
# healthy run or a slow model - it only trips on the collapse.
STALL_WINDOW_MINUTES = 20
STALL_MIN_COMPLETIONS = 5

# The first N rows of each CSV, in file order.
DEFAULT_LIMIT = 100


def checkpoint_path(category: Category, provider: config.Provider) -> Path:
    """Where one (category, provider) run records the cases it has finished."""
    return (
        config.variant_dir(VARIANT)
        / "checkpoints"
        / f"{category.name}_{config.short_name(provider)}.jsonl"
    )


class Checkpoint:
    """Append-only log of finished cases for one (category, provider) run.

    Written a line at a time and flushed, so a killed run loses at most the cases in
    flight. Reading it back is what makes a restart skip work already paid for.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.done: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A half-written last line from a killed run. Dropping it means
                    # that one case is redone, which is cheaper than guessing.
                    continue
                self.done[str(record["case_id"])] = record

    def record(self, case_id: str, classification: dict, usage: dict) -> None:
        entry = {"case_id": case_id, "classification": classification, "usage": usage}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
                handle.flush()
            self.done[case_id] = entry


class Progress:
    """Thread-safe reporting across every concurrent run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started = time.time()

    def say(self, message: str) -> None:
        with self._lock:
            elapsed = time.time() - self.started
            print(f"[{elapsed / 60:6.1f}m] {message}", flush=True)


class Heartbeat:
    """Completion rate across the whole sweep, and whether it has collapsed.

    Rate, not silence. The first version of this tripped only when no case finished
    for 25 minutes, and it missed the throttle it was written for: degraded runs still
    land a case every few minutes, which reset the timer forever. It fired after 528
    minutes instead of 70. Counting completions in a rolling window catches the case
    that actually happens - a sweep still moving, but two orders of magnitude too slow.

    The margin is wide on purpose. Healthy throughput measured here is ~7 cases/minute
    across four models; throttled is ~0.06. A floor of a few cases per window sits far
    below anything healthy and far above a true stall, so neither a slow model nor a
    long retry can trip it.
    """

    def __init__(self, window_minutes: float, min_completions: int) -> None:
        self._lock = threading.Lock()
        self._completions: list[float] = []
        self.window_seconds = window_minutes * 60
        self.min_completions = min_completions
        self.started = time.time()
        self.stalled = threading.Event()

    def beat(self) -> None:
        with self._lock:
            self._completions.append(time.time())

    def recent(self) -> int:
        """Cases completed within the trailing window."""
        cutoff = time.time() - self.window_seconds
        with self._lock:
            self._completions = [t for t in self._completions if t >= cutoff]
            return len(self._completions)

    def watch(self, progress: Progress, done: threading.Event) -> None:
        """Poll until the sweep finishes or its completion rate collapses."""
        while not done.wait(60):
            # Only judge a full window, so startup is not mistaken for a stall.
            if time.time() - self.started < self.window_seconds:
                continue
            recent = self.recent()
            if recent < self.min_completions:
                self.stalled.set()
                progress.say(
                    f"STALLED - only {recent} case(s) completed in the last "
                    f"{self.window_seconds / 60:.0f} minutes, below the floor of "
                    f"{self.min_completions}. The endpoint is throttling. Stopping; "
                    f"finished cases are checkpointed, so rerunning resumes."
                )
                return


class PairResult:
    """What one (category, provider) run produced."""

    def __init__(self, category: Category, provider: config.Provider) -> None:
        self.category = category
        self.provider = provider
        self.usage = Usage()
        self.failures: list[tuple[str, str]] = []
        self.n_cases = 0
        self.n_labelled = 0
        self.seconds = 0.0
        self.aborted: str | None = None


def run_pair(
    category: Category,
    provider: config.Provider,
    settings: config.Settings,
    limit: int | None,
    concurrency: int,
    progress: Progress,
    heartbeat: Heartbeat,
    max_attempts: int = MAX_ATTEMPTS,
) -> PairResult:
    """Classify one category with one provider, `concurrency` cases at a time."""
    result = PairResult(category, provider)
    started = time.time()
    label = f"{config.short_name(provider):8s} {category.name}"

    prompt = load_prompt(config.prompt_path(category, VARIANT))
    cases = read_cases(config.input_csv_path(category), category)[:limit]
    result.n_cases = len(cases)

    checkpoint = Checkpoint(checkpoint_path(category, provider))
    todo = [c for c in cases if c.case_id not in checkpoint.done]
    if len(todo) < len(cases):
        progress.say(f"{label}: resuming, {len(cases) - len(todo)} already done")

    # One client per pair, shared by the threads: httpx's client is thread-safe and
    # a client per thread pointed at the same endpoint would buy nothing.
    model = build_model(provider, settings)
    counter = {"n": len(cases) - len(todo)}
    counter_lock = threading.Lock()
    abort = threading.Event()

    def do_case(case) -> None:
        if abort.is_set() or heartbeat.stalled.is_set():
            return
        try:
            outcome = classify_case(model, prompt, category, case, max_attempts)
        except ProviderUnavailable as error:
            # The key or the subscription is the problem; every remaining case would
            # fail the same way, so stop this provider rather than burn the queue.
            abort.set()
            result.aborted = str(error)
            progress.say(f"{label}: ABORTED - {error}")
            return
        except Exception as error:  # noqa: BLE001 - one bad case must not end the run
            result.failures.append((case.case_id, f"{type(error).__name__}: {error}"))
            progress.say(
                f"{label}: case {case.case_id} FAILED - {type(error).__name__}"
            )
            return

        checkpoint.record(
            case.case_id,
            outcome.classification.model_dump(mode="json"),
            outcome.usage.model_dump(mode="json"),
        )
        heartbeat.beat()
        with counter_lock:
            counter["n"] += 1
            n = counter["n"]
        if n % 10 == 0 or n == len(cases):
            progress.say(f"{label}: {n}/{len(cases)}")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(do_case, todo))

    # Rebuild the outputs from the checkpoint, so a resumed run writes exactly what a
    # single uninterrupted run would have written.
    labels_by_case: dict[str, dict[str, str]] = {}
    reasoning: list[dict] = []
    for case in cases:
        record = checkpoint.done.get(case.case_id)
        if record is None:
            continue
        classification = record["classification"]
        labels = {item["finding"]: item["label"] for item in classification["findings"]}
        labels_by_case[case.case_id] = build_label_row(category, labels)
        reasoning.append(classification)
        result.usage = result.usage + Usage(**record["usage"])

    predictions = config.predictions_path(category, provider, VARIANT)
    write_labeled_csv(
        config.input_csv_path(category), predictions, category, labels_by_case
    )
    reasoning_file = config.reasoning_path(category, provider, VARIANT)
    reasoning_file.parent.mkdir(parents=True, exist_ok=True)
    reasoning_file.write_text(json.dumps(reasoning, indent=2), encoding="utf-8")

    result.n_labelled = len(labels_by_case)
    result.seconds = time.time() - started
    progress.say(
        f"{label}: done {result.n_labelled}/{len(cases)} in "
        f"{result.seconds / 60:.1f}m -> {predictions.name}"
    )
    return result


def run_provider(
    provider: config.Provider,
    categories: tuple[Category, ...],
    settings: config.Settings,
    limit: int | None,
    concurrency: int,
    progress: Progress,
    heartbeat: Heartbeat,
    max_attempts: int = MAX_ATTEMPTS,
) -> list[PairResult]:
    """One provider's whole share of the sweep, category after category."""
    results: list[PairResult] = []
    for category in categories:
        results.append(
            run_pair(
                category,
                provider,
                settings,
                limit,
                concurrency,
                progress,
                heartbeat,
                max_attempts,
            )
        )
        if results[-1].aborted:
            progress.say(f"{config.short_name(provider)}: skipping remaining categories")
            break
        if heartbeat.stalled.is_set():
            break
    return results


def write_token_log(results: list[PairResult], limit: int) -> Path:
    """Fill in the token log config declares but nothing has been writing."""
    path = config.LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    scope = "all cases" if limit <= 0 else f"first {limit} cases"
    lines = [
        "# Token usage - classifier_multi few-shot sweep",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}. "
        f"Variant `{VARIANT}`, {scope} of each category.",
        "",
        "| category | model | cases | input | output | total | minutes | failures |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    total = Usage()
    for r in sorted(results, key=lambda r: (r.category.name, r.provider)):
        total = total + r.usage
        lines.append(
            f"| {r.category.name} | {config.short_name(r.provider)} "
            f"| {r.n_labelled}/{r.n_cases} "
            f"| {r.usage.input_tokens:,} | {r.usage.output_tokens:,} "
            f"| {r.usage.total_tokens:,} | {r.seconds / 60:.1f} "
            f"| {len(r.failures)} |"
        )
    lines += [
        f"| **all** | | | **{total.input_tokens:,}** | **{total.output_tokens:,}** "
        f"| **{total.total_tokens:,}** | | |",
        "",
    ]

    by_model: dict[str, Usage] = {}
    for r in results:
        name = config.short_name(r.provider)
        by_model[name] = by_model.get(name, Usage()) + r.usage
    lines += ["## Per model", "", "| model | total tokens |", "| --- | ---: |"]
    lines += [
        f"| {name} | {usage.total_tokens:,} |" for name, usage in sorted(by_model.items())
    ]
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classifier_multi.run_fewshot",
        description="Few-shot sweep: every category, every model, in parallel.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"first N cases of each CSV, in file order "
        f"(default {DEFAULT_LIMIT}; 0 means all)",
    )
    parser.add_argument(
        "--categories", nargs="+", default=list(CATEGORY_NAMES), choices=CATEGORY_NAMES
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=list(config.PROVIDERS),
        choices=config.PROVIDERS,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=PER_MODEL_CONCURRENCY,
        help=f"requests in flight per model (default {PER_MODEL_CONCURRENCY}, below "
        f"the burst ceiling of 3 because sustained load gets throttled)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_ATTEMPTS,
        help=f"attempts per case before giving up on it (default {MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--stall-window",
        type=float,
        default=STALL_WINDOW_MINUTES,
        help=f"minutes of throughput to judge (default {STALL_WINDOW_MINUTES}; "
        f"0 disables the watchdog)",
    )
    parser.add_argument(
        "--stall-floor",
        type=int,
        default=STALL_MIN_COMPLETIONS,
        help=f"stop if fewer than this many cases finish per window "
        f"(default {STALL_MIN_COMPLETIONS})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = config.load_settings()
    categories = tuple(get_category(name) for name in args.categories)
    providers = tuple(args.providers)
    limit = args.limit if args.limit > 0 else None

    progress = Progress()
    n_cases = sum(
        len(read_cases(config.input_csv_path(c), c)[:limit]) for c in categories
    )
    progress.say(
        f"{len(providers)} models x {n_cases} cases = "
        f"{len(providers) * n_cases} calls, {args.concurrency} in flight per model"
    )

    # An infinite window disables the watchdog: the thread still runs and still exits
    # with the sweep, it just never judges a window.
    heartbeat = Heartbeat(
        args.stall_window if args.stall_window > 0 else float("inf"),
        args.stall_floor,
    )
    finished = threading.Event()
    watchdog = threading.Thread(
        target=heartbeat.watch, args=(progress, finished), daemon=True
    )
    watchdog.start()

    results: list[PairResult] = []
    try:
        with ThreadPoolExecutor(max_workers=len(providers)) as pool:
            for pair_results in pool.map(
                lambda p: run_provider(
                    p,
                    categories,
                    settings,
                    limit,
                    args.concurrency,
                    progress,
                    heartbeat,
                    args.max_attempts,
                ),
                providers,
            ):
                results.extend(pair_results)
    finally:
        finished.set()

    log = write_token_log(results, args.limit)
    progress.say(f"token log -> {log}")

    failures = [(r, cid, msg) for r in results for cid, msg in r.failures]
    aborted = [r for r in results if r.aborted]
    if failures:
        progress.say(f"{len(failures)} case(s) failed after all retries:")
        for r, cid, msg in failures:
            progress.say(
                f"  {r.category.name} {config.short_name(r.provider)} {cid}: {msg}"
            )
    if aborted:
        for r in aborted:
            progress.say(f"{config.short_name(r.provider)} aborted: {r.aborted}")
        return 1
    if heartbeat.stalled.is_set():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
