"""BCP reference using mini's real Drop warmup and Reposition sequence protocol.

Runs the production tokenizer sequence and scheduler in the existing resident
model pump. No dense-attention replacement and no change to mini's source.
"""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from bcp_numeric_fixture import load_fixture, request_for


async def main():
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    root = Path(os.environ["CONTEXT_MINI_ROOT"])
    sys.path.insert(0, str(root / "python"))
    spec = importlib.util.spec_from_file_location(
        "mini_staged_runner", root / "tests/contextual/mask_staged_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from minisgl.message import RepositionOpenAckMsg, WarmupAckMsg
    from minisgl.tokenizer.reposition_sequence import RepositionSequenceState

    original_config = module.SchedulerConfig

    def config(**kwargs):
        kwargs.update(
            tp_info=module.DistributedInfo(
                int(os.environ.get("LOCAL_RANK", "0")),
                int(os.environ.get("WORLD_SIZE", "1")),
            ),
            contextual_prefill_mode="staged",
            reposition_execution_mode="staged",
            max_extend_tokens=int(os.environ.get("CONTEXT_CHUNK_SIZE", "8192")),
            max_seq_len_override=int(os.environ.get("CONTEXT_MAX_LENGTH", "24576")),
            num_page_override=int(os.environ.get("CONTEXT_KV_CAPACITY", "24576")),
        )
        return original_config(**kwargs)

    module.SchedulerConfig = config
    original_body = module.request_body

    def body(*args, **kwargs):
        kwargs["ignore_eos"] = True
        return original_body(*args, **kwargs)

    module.request_body = body
    model = os.environ["CONTEXT_SERVER_MODEL"]
    runner = module.Runner(model, reference_alignment=True)
    sequences, pending = {}, []
    original_build, original_reply = module._build_user_msg, runner.reply

    def build(msg, tokenized):
        if tokenized.reposition_input_ids is None:
            return original_build(msg, tokenized)
        state = RepositionSequenceState.pending(msg, tokenized)
        sequences[msg.uid] = state
        return state.open_msg()

    def reply(messages):
        public = []
        for message in messages:
            state = sequences.get(message.uid)
            if isinstance(message, RepositionOpenAckMsg):
                assert state is not None
                state.activate(step_token_budget=message.step_token_budget)
            elif isinstance(message, WarmupAckMsg) and state is not None:
                state.accept_ack(message)
            else:
                public.append(message)
                continue
            pending.append(state.build_next_msg())
            if state.in_flight_final:
                sequences.pop(message.uid)
        if public:
            original_reply(public)

    module._build_user_msg = build
    runner.scheduler.send_result = reply
    schedule = runner.scheduler._schedule_next_batch

    def next_batch():
        while pending:
            message = pending.pop(0)
            runner.scheduler._process_one_msg(
                module.BaseBackendMsg.decoder(module.BaseBackendMsg.encoder(message))
            )
        return schedule()

    runner.scheduler._schedule_next_batch = next_batch
    sampler = runner.scheduler.engine.sampler
    observed_sample = sampler.sample

    def raw_sample(logits, args):
        raw = logits.detach().clone()
        result = observed_sample(logits, args)
        for i, req in enumerate(runner.batch.reqs):
            record = runner.records[req.uid]
            if req.sample_is_committed and not req.is_warmup:
                record["logits_gpu"][-1] = raw[i]
            else:
                record["logits_gpu"].pop()
                record["sample_rows"].pop()
        return result

    sampler.sample = raw_sample
    export = runner.export_logits

    def export_logits():
        for record in runner.records.values():
            if record.get("logits_gpu") == []:
                del record["logits_gpu"]
        export()

    runner.export_logits = export_logits
    fixture = load_fixture()
    if trace := fixture.get("native_sglang_trace"):
        assert runner.tokenizer.is_gpt_oss and trace["model"] == model
        expected = [module.api.Message(**m).model_dump() for m in fixture["messages"]]

        def normalized_harmony(messages, **kwargs):
            assert messages == expected
            runner.tokenizer._tokenize_invocations += 1
            runner.tokenizer._chat_template_invocations += 1
            runner.tokenizer._harmony_thinking_ranges = {}
            return (
                list(trace["input_ids"]),
                list(trace["owners"]),
                trace["generation_start"],
            )

        # Only full-history Reposition can consume this frozen-token adapter.
        # Drop prefix warmups must use mini's actual prefix template renderer.
        native_harmony = runner.tokenizer._render_harmony_message_drop

        def harmony(messages, **kwargs):
            return (
                normalized_harmony(messages, **kwargs)
                if messages == expected
                else native_harmony(messages, **kwargs)
            )

        runner.tokenizer._render_harmony_message_drop = harmony
    result = {
        "model": model,
        "fixture": fixture,
        "reference_mode": "staged",
        "mini_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "runs": {},
    }
    logits = {}

    async def run(feature, key, request=None, tokens=None):
        runner.forced_tokens = tokens
        count = (
            len(tokens)
            if tokens
            else int(os.environ.get("CONTEXT_NUMERIC_TOKENS", "8"))
        )
        value = await runner.generate(
            "staged", [request or request_for(fixture, feature)], max_tokens=count
        )
        assert not sequences and not pending
        formal = [r for r in value["records"] if not r["warmup"]]
        assert len(formal) == 1 and formal[0]["tokens"]
        if tokens is not None:
            assert formal[0]["tokens"] == tokens, key
        assert len(runner.full_logits[formal[0]["uid"]]) == len(
            formal[0]["tokens"]
        ), key
        value["warmup_records"] = [r for r in value["records"] if r["warmup"]]
        value["records"] = formal
        result["runs"][key] = value
        logits[key] = runner.full_logits[formal[0]["uid"]]
        print(
            "MINI_STAGED_BCP",
            key,
            len(formal[0]["stages"]),
            formal[0]["tokens"],
            flush=True,
        )
        return formal[0]["tokens"]

    for feature in ("none", "drop", "drop_repos"):
        runner.clear()
        await run(feature, feature)
    tokens = result["runs"]["drop_repos"]["records"][0]["tokens"]
    await run("drop_repos", "drop_repos-hot", tokens=tokens)
    runner.clear()
    await run(
        "none",
        "none-retry-source",
        tokens=result["runs"]["none"]["records"][0]["tokens"],
    )
    await run("drop_repos", "drop_repos-retry", tokens=tokens)
    runner.clear()
    source = request_for(fixture, "drop_repos")
    source["reposition"] = fixture["reposition"][:1]
    await run("drop_repos", "consecutive-source", source, tokens[:1])
    await run("drop_repos", "drop_repos-consecutive", tokens=tokens)
    runner.clear()
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        output = os.environ["CONTEXT_ORACLE_OUTPUT"]
        Path(output).write_text(json.dumps(result))
        module.torch.save(logits, output + ".pt")


if __name__ == "__main__":
    asyncio.run(main())
