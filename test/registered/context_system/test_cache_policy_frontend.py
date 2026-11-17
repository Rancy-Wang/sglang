"""String interface, legacy isolation, and validated transport boundaries."""

from types import SimpleNamespace

import pytest
import torch
from test_cache_policy_ir import POLICIES, compile_policy
from test_ir import ROOT, load_file

pytest_plugins = ("test_cache_policy_ir",)


@pytest.mark.parametrize("policy", POLICIES)
def test_string_interface_and_invalid_payloads(policy_modules, policy):
    rules, _, planner = policy_modules
    messages = [{"role": "user", "content": "x"}] * 4
    rule = rules.parse_drop_rule(policy, messages, legacy_drop_message={"2": [1]})
    assert rule.type == policy and rule.drop_messages == {2: (1,)}
    assert rules.parse_drop_rule(policy, messages).drop_messages == {}
    with pytest.raises(ValueError):
        rules.parse_drop_rule(policy, messages, legacy_drop_message={"1": [2]})
    with pytest.raises(ValueError):
        rules.parse_drop_rule(
            {"type": "message_drop", "drop_messages": {}},
            messages,
            legacy_drop_message={},
        )
    trace = SimpleNamespace(input_ids=[1, 2, 3, 4], owners=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="do not support Insert"):
        planner.compile_chat_program(messages, trace, rule, [], {3: -1})


def test_legacy_wire_unchanged(policy_modules):
    rules, _, planner = policy_modules
    messages = [{"role": "user", "content": "x"}] * 4
    trace = SimpleNamespace(input_ids=[1, 2, 3, 4], owners=[0, 1, 2, 3])
    p = planner.compile_chat_program(
        messages,
        trace,
        rules.parse_drop_rule(None, messages, legacy_drop_message={"2": [1]}),
        [3],
    )
    assert p.to_wire()["version"] == 3
    assert p.to_json_wire()["version"] == 4
    assert "cache_policy" not in p.to_wire()["layout"]
    assert set(p.to_wire()) == {"version", "layout", "visible_until"}
    restored = planner.ContextProgram.from_wire(p.to_wire(), p.input_ids)
    assert restored.layout.cache_policy == 0 and restored.allow_retry


@pytest.mark.parametrize("policy", POLICIES)
def test_new_wire_rejects_policy_or_compute_identity_corruption(policy_modules, policy):
    p = compile_policy(policy_modules, policy, [1, 2, 3, 4], {3: [(1, 2)]})
    planner = policy_modules[2]
    wire = p.to_wire()
    wire["layout"] = {**wire["layout"], "cache_policy": 3}
    with pytest.raises(ValueError, match="Invalid cache policy"):
        planner.ContextProgram.from_wire(wire, p.input_ids)
    wire = p.to_wire()
    wire["execution_to_full"] = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="provenance"):
        planner.ContextProgram.from_wire(wire, p.input_ids)
    wire = p.to_wire()
    wire["layout"] = {**wire["layout"], "birth_positions": p.layout.birth_positions + 1}
    with pytest.raises(ValueError, match="compute identity|forbids rotations"):
        planner.ContextProgram.from_wire(wire, p.input_ids)


def test_hybrid_rejection_is_new_policy_only():
    admission = load_file(
        "cache_policy_admission",
        ROOT / "python/sglang/srt/context_system/capabilities.py",
    )
    config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Qwen3_5ForCausalLM"])
    )
    args = SimpleNamespace(page_size=1)
    with pytest.raises(ValueError, match="New cache policies do not support"):
        admission.validate_context_config(args, config, cache_policy=POLICIES[0])


@pytest.mark.skipif(
    __import__("sys").platform != "linux", reason="native SRT frontend requires Linux"
)
@pytest.mark.parametrize("policy", POLICIES)
def test_public_pydantic_schema(policy):
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "x", "index": 0}],
        drop_rule=policy,
        drop_message={},
    )
    assert request.drop_rule == policy
    assert request.context_index_metadata()["context_next_message_index"] == 1
