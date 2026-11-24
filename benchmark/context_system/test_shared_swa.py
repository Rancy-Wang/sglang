import pytest

from shared_swa import FLAG, shared_swa_command, validate_shared_swa


def test_launch_policy_preserves_other_models_and_deduplicates():
    native = ["python", "-m", "sglang.launch_server", "--model-path", "Qwen3-8B"]
    assert shared_swa_command(native) == native
    command = native[:-1] + ["gpt-oss-120b"]
    assert shared_swa_command(command) == command + [FLAG]
    assert shared_swa_command(command + [FLAG]) == command + [FLAG]
    assert FLAG not in command


@pytest.mark.parametrize("flag", [None, False, "true", 1])
def test_external_server_must_report_effective_shared_policy(flag):
    with pytest.raises(ValueError, match=FLAG):
        validate_shared_swa({"model_path": "gpt-oss-120b", "disable_hybrid_swa_memory": flag}, "alias")


def test_effective_policy_accepts_shared_and_non_gpt_models():
    validate_shared_swa({"disable_hybrid_swa_memory": True}, "gpt-oss-120b")
    validate_shared_swa({"server_args": {"disable_hybrid_swa_memory": True}}, "gpt-oss-120b")
    validate_shared_swa({"disable_hybrid_swa_memory": False}, "Qwen3-8B")
