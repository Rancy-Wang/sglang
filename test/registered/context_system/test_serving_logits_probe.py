import sys
from types import SimpleNamespace

import pytest
import torch

from serving_logits_probe import serialized_probe

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="native SRT runtime requires Linux"
)


def test_probe_excludes_chunk_samples_and_exports_original_fixed_logits(tmp_path):
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

    processor = CustomLogitProcessor.from_str(serialized_probe())
    req = SimpleNamespace(
        origin_input_ids=[1, 2, 3], extend_range=SimpleNamespace(end=2)
    )
    path = tmp_path / "logits.pt"
    params = {
        "__req__": req,
        "context_trace_path": str(path),
        "context_trace_count": 2,
        "context_forced_tokens": [1, 0],
    }
    first = torch.tensor([[3.0, 2.0, 1.0]])
    torch.testing.assert_close(processor(first.clone(), [params]), first)
    assert not path.exists()
    req.extend_range.end = 3
    assert processor(first.clone(), [params]).argmax().item() == 1
    assert not path.exists()
    second = first + 0.5
    assert processor(second.clone(), [params]).argmax().item() == 0
    torch.testing.assert_close(
        torch.load(path, weights_only=True), torch.cat([first, second])
    )
    assert not processor.rows
