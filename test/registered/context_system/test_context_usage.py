import pytest
import torch
from test_ir import ROOT, load_file


@pytest.fixture
def usage_type():
    return load_file(
        "context_usage_under_test", ROOT / "python/sglang/srt/context_system/usage.py"
    ).ContextUsage


def mask(*values):
    return torch.tensor(values, dtype=torch.bool)


def test_read_union_excludes_holes_swa_trimming_and_cacheback_copies(usage_type):
    usage = usage_type(mask(1, 1, 1, 1, 0, 1), mask(1, 1, 0, 0, 0, 1))
    # Raw 0: dropped and never read. Raw 1: read before its drop. Raw 2:
    # unused but not dropped (e.g. an SWA-only diagnostic), cannot count as skip.
    # Raw 4: a hole, never a cache hit. Raw 5: cacheback copied but never read.
    usage.record_prefill(mask(0, 1, 0, 1, 1, 0), mask(0, 0, 0, 0, 1, 1), 3)
    usage.record_prefill(mask(0, 0, 0, 1, 0, 0), mask(0, 0, 0, 1, 0, 0), 2)
    usage.record_decode(2)
    result = usage.snapshot()
    assert (result.cached_tokens, result.repos_tokens, result.drop_skipped_tokens) == (
        1,
        1,
        2,
    )
    assert (result.actual_prefill_tokens, result.actual_decode_tokens) == (5, 2)
    # Recomputing queries costs compute again, while cached read unions do not
    # double-count the same cached token across chunks or recovery passes.
    usage.record_prefill(mask(0, 1, 0, 1, 0, 0), mask(0, 0, 0, 0, 0, 0), 2)
    result = usage.snapshot()
    assert (result.cached_tokens, result.repos_tokens) == (1, 1)
    assert result.actual_prefill_tokens == 7


def test_usage_input_ownership_and_validation(usage_type):
    resident = mask(1, 1)
    usage = usage_type(resident, mask(0, 1))
    resident.zero_()
    usage.record_prefill(mask(1, 0), mask(0, 0), 1)
    assert usage.snapshot().cached_tokens == 1
    with pytest.raises(ValueError, match="initial match"):
        usage.record_prefill(mask(1), mask(0), 1)
    with pytest.raises(ValueError, match="actual model queries"):
        usage.record_decode(True)
