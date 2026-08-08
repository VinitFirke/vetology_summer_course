"""Makes the project root importable so tests can `from classifier import ...`."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """Skip the real retry backoff in every test - we test retry logic, not the clock.

    classify_case retries 10 times with backoff capped at 60s, so one unexpected failure
    inside a test sleeps for about seven minutes before reporting. Patching here rather
    than per-file means a new test module cannot forget it and hang the suite.
    """
    monkeypatch.setattr("classifier.classify.time.sleep", lambda _seconds: None)
