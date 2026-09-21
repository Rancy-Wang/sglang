"""Run separately with D Radix off/on; neither mode implies coverage of the other."""

import os
import pytest
from minimax_context_fixture import (
    endpoint_from_env,
    short_request,
    context_usage,
    run_swe_task,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("MINIMAX_DECODE_URL"),
    reason="explicit MiniMax P/D servers required",
)


@pytest.mark.parametrize("policy", ["native", "drop", "repos", "combined"])
def test_minimax_pd_repeated_requests(policy):
    assert os.environ.get("MINIMAX_DECODE_RADIX") in (
        "0",
        "1",
    ), "record the actual D Radix mode"
    endpoint = endpoint_from_env("pd")
    payload = short_request(policy)
    for i in range(2):
        result = endpoint.request(payload, policy + "-" + str(i))
        assert "M15" in result["choices"][0]["message"]["content"]
        if policy != "native":
            assert context_usage(result) is not None


@pytest.mark.skipif(
    not os.environ.get("MINIMAX_SWE_TASK"),
    reason="explicit SWE task and image required",
)
def test_minimax_pd_swe_rolling_drop_96k():
    assert os.environ.get("MINIMAX_DECODE_RADIX") in ("0", "1")
    report = run_swe_task(
        endpoint_from_env("pd"),
        os.environ["MINIMAX_SWE_TASK"],
        os.environ["MINIMAX_SWE_IMAGE"],
    )
    assert report[
        "coverage_complete"
    ], "SWE completed but did not exercise both rolling Drop and 96K Reposition"


@pytest.mark.skipif(
    not os.environ.get("MINIMAX_NUMERIC_REFERENCE"),
    reason="staged native reference required",
)
def test_minimax_pd_fixed_tokens_and_concurrency():
    from minimax_context_fixture import run_fixed_token_http

    assert os.environ.get("MINIMAX_DECODE_RADIX") in ("0", "1")
    run_fixed_token_http(
        endpoint_from_env("pd"), os.environ["MINIMAX_NUMERIC_REFERENCE"]
    )
