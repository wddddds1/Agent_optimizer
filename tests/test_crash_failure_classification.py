from orchestrator.graph import _is_crash_like_failure


def test_is_crash_like_failure_with_negative_exit_code() -> None:
    assert _is_crash_like_failure("nonzero exit code: -11", -11)


def test_is_crash_like_failure_with_signal_text() -> None:
    assert _is_crash_like_failure("runtime error: segmentation fault", 1)


def test_is_crash_like_failure_false_for_regular_nonzero() -> None:
    assert not _is_crash_like_failure("nonzero exit code: 1", 1)
