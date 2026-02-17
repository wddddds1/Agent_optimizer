from orchestrator.graph import _deep_analysis_next_step
from schemas.opportunity_graph import OpportunityStatus


def test_need_more_context_retries_once_then_errors() -> None:
    assert (
        _deep_analysis_next_step(
            OpportunityStatus.NEED_MORE_CONTEXT, retry_count=0, max_context_retries=1
        )
        == "RETRY_CONTEXT"
    )
    assert (
        _deep_analysis_next_step(
            OpportunityStatus.NEED_MORE_CONTEXT, retry_count=1, max_context_retries=1
        )
        == "ERROR_NEED_MORE_CONTEXT"
    )


def test_need_more_profile_is_explicit_error() -> None:
    assert (
        _deep_analysis_next_step(
            OpportunityStatus.NEED_MORE_PROFILE, retry_count=0, max_context_retries=1
        )
        == "ERROR_NEED_MORE_PROFILE"
    )


def test_no_actionable_is_explicit_error_and_ok_proceeds() -> None:
    assert (
        _deep_analysis_next_step(
            OpportunityStatus.NO_ACTIONABLE, retry_count=0, max_context_retries=1
        )
        == "ERROR_NO_ACTIONABLE"
    )
    assert (
        _deep_analysis_next_step(
            OpportunityStatus.OK, retry_count=0, max_context_retries=1
        )
        == "PROCEED"
    )
