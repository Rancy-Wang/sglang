"""Opt-in GPU check of P capacity rejection and D receive-queue cleanup."""

import concurrent.futures
import json
import os
import time
from pathlib import Path

import pytest

requests = pytest.importorskip("requests")
torch = pytest.importorskip("torch")
from test_bcp_pd_numeric import pd_servers  # noqa: F401

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONTEXT_PD_CAPACITY_PRESSURE"),
    reason="explicit isolated PD capacity check required",
)


def test_capacity_rejection_releases_both_sides(pd_servers):
    from sglang.srt.context_system.ir import (
        compile_context_layout,
        prewarm_context_layout,
    )
    from sglang.srt.context_system.planner import ContextProgram

    assert os.environ["CONTEXT_P_KV"] == "110"
    # Keep native D's 512-token decode reserve. Its pool must admit the
    # 90-token recovery request as well as the rejected request's active KV.
    assert os.environ["CONTEXT_D_KV"] == "640"
    assert os.environ["CONTEXT_CHUNK_SIZE"] == "32"
    p_base, d_base, bootstrap = pd_servers
    prewarm_context_layout()
    tokens = [785] * 100
    layout = compile_context_layout(
        *(
            torch.tensor(value, dtype=torch.int32)
            for value in (tokens, [80], [0, 1], [0, 40], [98], [99])
        )
    )
    expiry = torch.full((100,), torch.iinfo(torch.int32).max, dtype=torch.int32)
    expiry[:40] = 80
    sampling = {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True}
    room = time.time_ns() % (1 << 52)
    evidence = {}
    output = Path(os.environ["CONTEXT_TRACE_DIR"]) / "capacity-result.json"

    def paired(name, payload):
        nonlocal room
        room += 1
        body = dict(
            payload,
            sampling_params=sampling,
            bootstrap_host="127.0.0.1",
            bootstrap_port=bootstrap,
            bootstrap_room=room,
        )

        def send(base):
            started = time.monotonic()
            response = requests.post(base + "/generate", json=body, timeout=60)
            return {
                "status": response.status_code,
                "seconds": time.monotonic() - started,
                "body": response.json(),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            replies = list(executor.map(send, (p_base, d_base)))
        evidence[name] = dict(zip(("prefill", "decode"), replies))
        output.write_text(json.dumps(evidence, indent=2))
        return replies

    p, d = paired(
        "rejected",
        {
            "input_ids": tokens,
            "context_program": ContextProgram(layout, expiry).to_json_wire(),
        },
    )
    assert p["status"] == 503 and "KV pool holds 110" in str(p["body"]), p
    # The native transport may surface its peer's failure as 500 or 503. It
    # must stop waiting rather than inventing a successful partial generation.
    assert d["status"] in (500, 503), d
    for base in (p_base, d_base):
        assert requests.post(base + "/flush_cache", timeout=5).status_code == 200
    p, d = paired("subsequent", {"input_ids": [785] * 90})
    assert p["status"] == d["status"] == 200, (p, d)
    assert d["body"]["meta_info"]["completion_tokens"] == 8, d
    for base in (p_base, d_base):
        assert requests.get(base + "/health", timeout=5).status_code == 200
