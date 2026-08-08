"""Reading a logprobs response. The network half is not tested; this half is.

Providers disagree about where logprobs live and what "absent" looks like, so the
inspection is a pure function and these cases are the shapes it has to survive.
"""

from uncertainty.probe import describe_logprobs


class Raw:
    def __init__(self, response_metadata):
        self.response_metadata = response_metadata


def test_openai_shape_is_recognised():
    raw = Raw({"logprobs": {"content": [{"token": "abnormal", "logprob": -0.03}]}})

    supported, detail = describe_logprobs(raw)

    assert supported is True
    assert "1 token" in detail


def test_a_bare_list_is_also_accepted():
    """Some OpenAI-compatible providers return the list without the content wrapper."""
    raw = Raw({"logprobs": [{"token": "normal", "logprob": -0.1}]})

    supported, _ = describe_logprobs(raw)

    assert supported is True


def test_a_missing_field_is_reported_not_raised():
    supported, detail = describe_logprobs(Raw({}))

    assert supported is False
    assert "no logprobs field" in detail


def test_an_explicit_null_is_treated_as_absent():
    """The common failure mode: the key is echoed back with nothing in it."""
    supported, detail = describe_logprobs(Raw({"logprobs": None}))

    assert supported is False
    assert "no logprobs field" in detail


def test_an_empty_content_list_is_not_usable():
    supported, detail = describe_logprobs(Raw({"logprobs": {"content": []}}))

    assert supported is False
    assert "empty" in detail


def test_an_unrecognised_shape_is_rejected_rather_than_half_accepted():
    raw = Raw({"logprobs": {"content": ["abnormal", "normal"]}})

    supported, detail = describe_logprobs(raw)

    assert supported is False
    assert "unrecognised shape" in detail


def test_a_raw_message_with_no_metadata_at_all_is_handled():
    class Bare:
        pass

    supported, _ = describe_logprobs(Bare())

    assert supported is False


def test_none_raw_is_handled():
    supported, _ = describe_logprobs(None)

    assert supported is False
