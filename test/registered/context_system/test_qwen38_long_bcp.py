"""Replay the three saved warm-cache crash cases against an isolated server.

Run directly with --mode fault or full. Inputs are preserved experiment artifacts;
no recorded final assistant answer is sent, and generated interim answers are not
substituted for the fixed trajectory. Full mode validates every tool turn and
saves a newly generated final answer for independent grading.
"""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path

import requests

from minimax_context_fixture import RollingDrop96K, output_findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/share/public/public_models/Qwen3.8-27B")
    parser.add_argument("--cases", default="775,844,792")
    parser.add_argument("--mode", choices=["fault", "full", "stress"], required=True)
    opt = parser.parse_args()
    opt.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False

    def flush():
        r = session.post(opt.url + "/flush_cache", timeout=30)
        r.raise_for_status()

    def send(case, label, payload):
        stem = opt.output / f"{case}-{label}"
        payload = copy.deepcopy(payload)
        payload.update(rid=f"r1-{case}-{label}-{time.time_ns()}", return_meta_info=True,
                       return_input_ids_in_sglext=True, return_output_ids_in_sglext=True)
        stem.with_suffix(".request.json").write_text(json.dumps(payload, ensure_ascii=False))
        start = time.monotonic()
        record = dict(case=case, label=label, rid=payload["rid"],
                      messages=len(payload["messages"]), repositions=payload.get("reposition", []))
        try:
            response = session.post(opt.url + "/v1/chat/completions", json=payload, timeout=1800)
            stem.with_suffix(".response.json").write_text(response.text)
            record["http_status"] = response.status_code
            response.raise_for_status()
            result = response.json()
            choice = result["choices"][0]
            record["meta"] = choice.get("meta_info", {})
            record["finish_reason"] = choice["finish_reason"]
            record["findings"] = output_findings(choice["message"])
            assert not record["findings"], record
            assert choice["message"].get("content") or choice["message"].get("reasoning_content") or choice["message"].get("tool_calls"), choice
            session.get(opt.url + "/health", timeout=20).raise_for_status()
            assert record["meta"].get("num_retractions", 0) == 0, record
            return result
        except Exception as exc:
            record["error"] = repr(exc)
            raise
        finally:
            record["seconds"] = time.monotonic() - start
            with (opt.output / "progress.jsonl").open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)

    if opt.mode == "stress":
        flush()
        pending = []
        for case, warm, target in [("775", 66, 68), ("792", 56, 58)]:
            full = json.loads((opt.inputs / f"case{case}" / "full-request.json").read_text())
            def prefix(index, count):
                return dict(full, model=opt.model, messages=full["messages"][:index+1], max_tokens=count,
                            drop_message={k:v for k,v in full["drop_message"].items() if int(k)<=index},
                            reposition=[i for i in full["reposition"] if i<=index])
            send(case, "interleave-warm", prefix(warm, 8))
            pending.append((case, prefix(target, 64)))
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(send, case, "interleave-reposition", payload) for case, payload in pending]
            for future in futures:
                result = future.result()
                assert result["choices"][0]["meta_info"]["cached_tokens"] > 0
        flush()
        case, payload = pending[0]
        payload = dict(payload, rid="r1-explicit-abort", max_tokens=4096)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(requests.post, opt.url + "/v1/chat/completions", json=payload, timeout=180)
            time.sleep(2)
            aborted = session.post(opt.url + "/abort_request", json={"rid": payload["rid"]}, timeout=30)
            aborted.raise_for_status()
            response = future.result()
            (opt.output / "explicit-abort.json").write_text(json.dumps({
                "request_id": payload["rid"], "abort_status": aborted.status_code,
                "response_status": response.status_code, "response": response.text,
            }, indent=2))
            assert "abort" in response.text.lower(), response.text
        send("stress", "after-abort", dict(model=opt.model,
             messages=[{"role":"user", "content":"Reply only OK."}], max_tokens=16,
             temperature=0, chat_template_kwargs={"enable_thinking":False}))
        flush()
        return

    for case in opt.cases.split(","):
        full = json.loads((opt.inputs / f"case{case}" / "full-request.json").read_text())
        full["model"] = opt.model
        if opt.mode == "fault":
            stages = {"775": [24, 66, 68], "844": [24, 52, 64, 66], "792": [24, 56, 58]}[case]
            # Repeat without restarting the server, rebuilding the same warm
            # source branch each time. Eight tokens exercises unfinished publish.
            for repeat in range(2):
                flush()
                for index in stages:
                    payload = dict(full, messages=full["messages"][:index+1], max_tokens=8,
                                   drop_message={k:v for k,v in full["drop_message"].items() if int(k)<=index},
                                   reposition=[i for i in full["reposition"] if i<=index])
                    result = send(case, f"fault{repeat}-m{index:03d}", payload)
                    if index == stages[-1]:
                        assert result["choices"][0]["meta_info"]["cached_tokens"] > 0, "cold fallback masks regression"
            if case == "792":
                for label, indices, use_repos in [("cold-combined", [58], True), ("drop-only", [24,56,58], False)]:
                    flush()
                    for index in indices:
                        payload = dict(full, messages=full["messages"][:index+1], max_tokens=8,
                                       drop_message={k:v for k,v in full["drop_message"].items() if int(k)<=index},
                                       reposition=[i for i in full["reposition"] if i<=index] if use_repos else [])
                        send(case, f"{label}-m{index:03d}", payload)
            continue

        from transformers import AutoTokenizer
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
        request = ChatCompletionRequest.model_validate(full)
        messages = [m.model_dump() for m in request.messages]
        tools = [t.model_dump() for t in request.tools]
        policy = RollingDrop96K(AutoTokenizer.from_pretrained(opt.model, local_files_only=True))
        options = policy.extend(messages, tools)
        (opt.output / f"{case}-canonical-policy.json").write_text(json.dumps({
            "counts": policy.last_counts, "checks": policy.checks, **options,
        }, indent=2))
        full.update(options)
        flush()
        for index, message in enumerate(full["messages"]):
            if message["role"] != "tool":
                continue
            final = index == len(full["messages"]) - 1
            payload = dict(full, messages=full["messages"][:index+1], max_tokens=8192 if final else 8,
                           drop_message={k:v for k,v in options["drop_message"].items() if int(k)<=index},
                           reposition=[i for i in options["reposition"] if i<=index])
            if final:
                payload["tool_choice"] = "none"
            result = send(case, f"full-m{index:03d}", payload)
            if final:
                (opt.output / f"{case}-final-answer.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
                assert result["choices"][0]["finish_reason"] == "stop", "final answer truncated"
        # An ordinary request after the trajectory checks continued service.
        send(case, "health-generation", dict(model=opt.model, messages=[{"role":"user", "content":"Reply only OK."}],
             max_tokens=16, temperature=0, chat_template_kwargs={"enable_thinking": False}))


if __name__ == "__main__":
    main()
