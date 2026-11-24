"""Launch-time policy for GPT-OSS experiments; no per-request work."""

import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, build_opener


FLAG = "--disable-hybrid-swa-memory"


def is_gpt_oss(model):
    if "gpt-oss" in str(model).lower():
        return True
    config = Path(model) / "config.json"
    if config.is_file():
        return "GptOssForCausalLM" in json.loads(config.read_text()).get("architectures", [])
    return False


def shared_swa_command(command):
    """Copy a launch command, preserving non-GPT models and native settings."""
    result = list(command)
    model = result[result.index("--model-path") + 1]
    if is_gpt_oss(model) and FLAG not in result:
        result.append(FLAG)
    return result


def validate_shared_swa(info, model):
    args = info.get("server_args", info)
    if (is_gpt_oss(model) or is_gpt_oss(args.get("model_path", ""))) and args.get(
        "disable_hybrid_swa_memory"
    ) is not True:
        raise ValueError(f"GPT-OSS experiments require {FLAG}; restart the server")
    return info


def check_shared_swa(url, model):
    """Check effective server configuration once, before warmup or timing."""
    parsed = urlsplit(url)
    endpoint = urlunsplit((parsed.scheme, parsed.netloc, "/server_info", "", ""))
    with build_opener(ProxyHandler({})).open(endpoint, timeout=30) as response:
        return validate_shared_swa(json.load(response), model)
