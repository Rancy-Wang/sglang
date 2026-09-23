"""Paired real-history fixtures for PLAN-CS-20260923-DECODE-BREAKDOWN-R1.

CPU only. Uses the pinned serving method's RollingState, native template and
source ownership; never obtains a Drop input by physically deleting text.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

PLAN = "PLAN-CS-20260923-DECODE-BREAKDOWN-R1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_fixture(fixture):
    rows = fixture["requests"]
    assert fixture["plan_id"] == PLAN
    assert len(rows) == fixture["batch_size"]
    assert len({r["case_id"] for r in rows}) == len(rows)
    assert len({r["rid"] for r in rows}) == len(rows)
    for r in rows:
        assert digest(r["input_ids"]) == r["input_sha256"]
        assert digest(r["messages"]) == r["messages_sha256"]
        assert digest(r["replay_tokens"]) == r["replay_sha256"]
        assert len(r["replay_tokens"]) >= fixture["max_new_tokens"] + 2
        assert 0 < r["drop_state"]["active_tokens"] < len(r["input_ids"])
        assert len(r["input_ids"]) + fixture["max_new_tokens"] <= 131072
        assert bool(r["drop_state"]["reposition"]) == fixture["require_repos"]
    return fixture


def paired_payloads(fixture, strategy):
    validate_fixture(fixture)
    if strategy not in ("drop", "no_drop"):
        raise ValueError(strategy)
    payloads = []
    for row in fixture["requests"]:
        payload = dict(rid=row["rid"], model=fixture["model"],
                       messages=copy.deepcopy(row["messages"]), tools=fixture["tools"],
                       chat_template_kwargs=fixture["template_kwargs"],
                       max_tokens=fixture["max_new_tokens"], ignore_eos=True,
                       temperature=0, stream=True, stream_options={"include_usage": True})
        if strategy == "drop":
            payload.update({k: row["drop_state"][k] for k in ("drop_message", "reposition")})
        payloads.append(payload)
    return payloads


def build(args):
    from test_serving import NativeTemplateAdapter, load_method

    method = load_method(args.mini_root)
    cases, manifest = method.load_cases(args.requests_path, 80, 42)
    renderer = NativeTemplateAdapter(args.model, {"preserve_thinking_history": True}, args.chat_template)
    tokenizer = renderer.renderer.tokenizer_manager.tokenizer
    candidates, audit = [], []
    # Select natural request boundaries. Historical assistant outputs remain
    # identical in both arms, rather than using separately generated datasets.
    for case in cases:
        turns = sorted(case["turns"], key=lambda t: abs((t.get("source_prompt_tokens") or 0) - args.target_tokens))
        for turn in turns:
            hint = turn.get("source_prompt_tokens")
            if hint and abs(hint - args.target_tokens) > args.tolerance:
                continue
            boundary = turn["source_assistant_message_id"]
            history = copy.deepcopy(case["trajectory"][:boundary])
            full, owners = renderer.render(history, manifest["tools"])
            if abs(full - args.target_tokens) > args.tolerance:
                continue
            state = method.RollingState(keep=12, threshold=96 * 1024).extend(history, owners, full)
            ratio = 1 - state["active_tokens"] / full
            audit.append(dict(case_id=case["case_id"], turn=turn["turn"], full=full,
                              active=state["active_tokens"], drop_ratio=ratio,
                              repositions=len(state["reposition"])))
            if ratio < args.min_drop_ratio or bool(state["reposition"]) != args.require_repos:
                continue
            # Teacher-forced continuation: concatenate recorded assistant text
            # from this point on. This is an explicit operator-control fixture,
            # not a natural generation or task-quality measurement.
            text = "\n".join(str(m.get(k) or "") for m in case["trajectory"][boundary:]
                             if m["role"] == "assistant" for k in ("reasoning_content", "content"))
            tokens = tokenizer.encode(text, add_special_tokens=False)[:args.max_new_tokens + 2]
            if len(tokens) < args.max_new_tokens + 2:
                continue
            ids = list(renderer.renderer.trace.input_ids)
            assert len(ids) == full
            candidates.append(dict(case_id=case["case_id"], turn=turn["turn"],
                rid=f"decode-breakdown-{len(candidates):02d}", messages=history,
                messages_sha256=digest(history), input_ids=ids, input_sha256=digest(ids),
                replay_tokens=tokens, replay_sha256=digest(tokens), drop_state=state,
                source_trajectory_sha256=case["trajectory_sha256"]))
            break
        if len(candidates) == args.batch_size:
            break
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".selection.json").write_text(json.dumps(audit, indent=2))
    if len(candidates) != args.batch_size:
        raise ValueError(f"Only {len(candidates)}/{args.batch_size} eligible distinct cases; see selection audit")
    fixture = dict(plan_id=PLAN, schema=1, batch_size=args.batch_size, model=args.model,
        tools=manifest["tools"], template_kwargs={"preserve_thinking_history": True},
        source=str(args.requests_path), seed=42, target_tokens=args.target_tokens,
        require_repos=args.require_repos, max_new_tokens=args.max_new_tokens,
        continuation_policy="teacher-forced concatenated recorded assistant text; not a quality test",
        requests=candidates)
    validate_fixture(fixture)
    with out.open("x") as f:
        json.dump(fixture, f, ensure_ascii=False)
    print(json.dumps(dict(output=str(out), batch_size=len(candidates),
        full_tokens=[len(r["input_ids"]) for r in candidates],
        active_tokens=[r["drop_state"]["active_tokens"] for r in candidates])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("mini-root", "requests-path", "model", "chat-template", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--target-tokens", type=int, default=49152)
    p.add_argument("--tolerance", type=int, default=4096)
    p.add_argument("--min-drop-ratio", type=float, default=.6)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--require-repos", action="store_true")
    build(p.parse_args())


if __name__ == "__main__":
    main()
