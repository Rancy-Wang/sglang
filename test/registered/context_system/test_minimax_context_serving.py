"""Run against an explicitly supplied ordinary MiniMax server, retaining evidence."""

import os
import pytest
from minimax_context_fixture import (
    endpoint_from_env,
    short_request,
    context_usage,
    run_swe_task,
)

requires_server = pytest.mark.skipif(
    not os.environ.get("MINIMAX_SERVER_URL"), reason="explicit MiniMax server required"
)


@requires_server
@pytest.mark.parametrize("policy", ["native", "drop", "repos", "combined"])
def test_minimax_ordinary_repeated_requests(policy):
    endpoint = endpoint_from_env("ordinary")
    payload = short_request(policy)
    outputs = [endpoint.request(payload, policy + "-" + str(i)) for i in range(2)]
    for result in outputs:
        assert "M15" in result["choices"][0]["message"]["content"]
        if policy != "native":
            assert context_usage(result) is not None
    if policy == "combined":
        assert context_usage(outputs[-1])["cached_tokens"] > 0


@requires_server
@pytest.mark.skipif(
    not os.environ.get("MINIMAX_SWE_TASK"),
    reason="explicit SWE task and image required",
)
def test_minimax_ordinary_swe_rolling_drop_96k():
    report = run_swe_task(
        endpoint_from_env("ordinary"),
        os.environ["MINIMAX_SWE_TASK"],
        os.environ["MINIMAX_SWE_IMAGE"],
    )
    assert report[
        "coverage_complete"
    ], "SWE completed but did not exercise both rolling Drop and 96K Reposition"


@pytest.mark.skipif(
    not os.environ.get("MINIMAX_NUMERIC_ARGV"),
    reason="isolated TP model oracle required",
)
def test_minimax_staged_native_numeric_oracle():
    from minimax_context_fixture import run_native_numeric_oracle

    run_native_numeric_oracle(
        os.environ["MINIMAX_NUMERIC_ARGV"], os.environ["MINIMAX_NUMERIC_OUTPUT"]
    )


@requires_server
@pytest.mark.skipif(
    not os.environ.get("MINIMAX_NUMERIC_REFERENCE"),
    reason="staged native reference required",
)
def test_minimax_ordinary_fixed_tokens_and_concurrency():
    from minimax_context_fixture import run_fixed_token_http

    run_fixed_token_http(
        endpoint_from_env("ordinary"), os.environ["MINIMAX_NUMERIC_REFERENCE"]
    )
