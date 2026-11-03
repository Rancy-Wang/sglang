"""Opt-in, read-only Insert cache audit; never enable in timing runs.

The reference walks every resident tree edge and compares individual record
triples. It deliberately does not use RadixKey.match, retry child buckets,
descendant bounds, or the production SWA selector. Supported scope is page-one
Full/Full+SWA Insert histories, whose SWA misses are repaired after matching.
"""

import hashlib
import json
from pathlib import Path


def token_units(key, *, depth=0, retry=False, count=None):
    """Independent real-token units, retaining every preceding virtual event."""
    context = key.context
    count = len(key) if count is None else min(count, len(key))
    if context is None:
        return [
            ((0, int(token), 0 if retry else depth + i),)
            for i, token in enumerate(key.token_ids[:count])
        ]
    result = []
    for raw in range(key.context_start, key.context_start + count):
        begin = context.token_to_record[raw - 1] + 1 if raw else 0
        end = context.token_to_record[raw] + 1
        unit = []
        for record in range(begin, end):
            kind, a, b = context.records[3 * record : 3 * record + 3]
            unit.append((kind, a, 0 if retry and kind == 0 else b))
        result.append(tuple(unit))
    return result


def common_units(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


def brute_frontier(root, key, *, retry=False):
    """Longest Full-resident compatible path; holes remain explicit proof edges."""
    wanted = token_units(key, retry=retry)
    best, visited = 0, 0
    stack = [(root, 0)]
    while stack:
        parent, depth = stack.pop()
        if depth == len(wanted):
            continue
        for child in parent.children.values():
            if (
                child.evicted
                or child.key.extra_key != key.extra_key
                or child.key.cache_salt != key.cache_salt
            ):
                continue
            visited += 1
            first = token_units(child.key, depth=depth, retry=retry, count=1)
            if not first or first[0] != wanted[depth]:
                continue
            units = token_units(child.key, depth=depth, retry=retry)
            matched = common_units(units, wanted[depth:])
            best = max(best, depth + matched)
            if matched == len(units):
                stack.append((child, depth + matched))
    return best, visited


def audit_match(cache, key, result, expected):
    """Verify returned slots/positions/residency against the chosen real path."""
    import torch
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    root = cache.tree_core.root_node
    node = cache.tree_core.node_by_id(result.last_device_node)
    path = []
    while node is not root:
        path.append(node)
        node = node.parent
    units, slots, positions, resident, swa = [], [], [], [], []
    for node in reversed(path):
        edge = token_units(node.key, depth=len(units))
        units.extend(edge)
        positions.extend(unit[-1][2] for unit in edge)
        slots.append(node.component_data[ComponentType.FULL].value)
        resident.extend([not node.context_hole] * len(edge))
        swa.extend(
            [node.component_data[ComponentType.SWA].value is not None] * len(edge)
        )
    count = len(result.device_indices)
    selected_exact = common_units(units, token_units(key))
    compatible = [
        [(kind, a, 0 if kind == 0 else b) for kind, a, b in unit] for unit in units
    ]
    target = [list(unit) for unit in token_units(key, retry=True)]
    checks = {
        "maximal_frontier": count == expected["retry"],
        "operation_identity": compatible == target[:count],
        "slot_identity": torch.equal(
            torch.cat(slots) if slots else result.device_indices[:0],
            result.device_indices,
        ),
        "position_identity": (
            result.context_source_positions is None
            if count == 0
            else positions == result.context_source_positions.tolist()
        ),
        "full_residency": (
            result.context_resident is None
            if count == 0
            else resident == result.context_resident.tolist()
        ),
        "swa_residency": result.context_swa_resident is None
        or swa == result.context_swa_resident.tolist(),
        "selected_exact": selected_exact == result.context_exact_prefix_len,
        "retry_flag": bool(result.context_retry) == (selected_exact < count),
    }
    return {
        "query_tokens": len(key),
        "matched": count,
        "selected_exact": selected_exact,
        "retry_only": count - selected_exact,
        "oracle_exact": expected["exact"],
        "oracle_retry": expected["retry"],
        "checks": checks,
        "pass": all(checks.values()),
    }


def create_cache_audit(config):
    """Native --forward-hooks factory; installs diagnostics only when requested."""
    import torch
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if getattr(UnifiedRadixCache, "_insert_state_audit", False):
        return
    UnifiedRadixCache._insert_state_audit = True
    native = UnifiedRadixCache.match_prefix
    native_finished = UnifiedRadixCache._cache_finished_req_native
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    output = Path(config["output"] + f".rank{rank}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    sequence = 0

    def emit(report):
        nonlocal sequence
        report["sequence"] = sequence
        sequence += 1
        with output.open("a") as target:
            target.write(json.dumps(report) + "\n")

    def match(cache, params):
        key = params.key
        enabled = (
            key.context is not None
            and key.context.has_insert
            and params.context_retry
            and not params.context_cache_publication
        )
        if not enabled:
            return native(cache, params)
        expected = {
            mode: brute_frontier(cache.tree_core.root_node, key, retry=mode == "retry")[
                0
            ]
            for mode in ("exact", "retry")
        }
        result = native(cache, params)
        report = audit_match(cache, key, result, expected)
        report.update(
            kind="match",
            rid=getattr(params.req, "rid", None),
            key_sha256=hashlib.sha256(key.context.records.tobytes()).hexdigest(),
        )
        emit(report)
        if not report["pass"] and config.get("fail_fast", True):
            raise AssertionError(f"Insert state audit failed: {report}")
        return result

    def finished(cache, req, *args, **kwargs):
        key = (
            req.make_prefix_key(req.origin_input_ids)
            if req.context_program is not None
            else None
        )
        result = native_finished(cache, req, *args, **kwargs)
        if key is not None and key.context is not None and key.context.has_insert:
            # Record the real publication frontier without asserting that every
            # old token must remain resident: eviction and a trailing Drop with
            # no computed successor can legitimately limit publication.
            emit(
                {
                    "kind": "publication",
                    "rid": req.rid,
                    "input_tokens": len(key),
                    "oracle_exact": brute_frontier(cache.tree_core.root_node, key)[0],
                    "oracle_retry": brute_frontier(
                        cache.tree_core.root_node, key, retry=True
                    )[0],
                }
            )
        return result

    UnifiedRadixCache.match_prefix = match
    UnifiedRadixCache._cache_finished_req_native = finished


def active_messages(payload):
    """Native rounding control only; never used as the Insert semantic oracle."""
    dropped = {i for values in payload.get("drop_message", {}).values() for i in values}
    result, ordinary = [], 0
    for message in payload["messages"]:
        index = message.get("index")
        if index is None:
            index = ordinary
        if index >= 0:
            ordinary += 1
        if index not in dropped:
            result.append({k: v for k, v in message.items() if k != "index"})
    return result


def brute_decode_reads(layout, generated, window=-1):
    """Independent scalar oracle, intentionally scanning the entire prompt."""
    n = len(layout.positions)
    order = layout.logical_order.tolist() or list(range(n))
    pairs = [(i, int(layout.positions[i])) for i in order if layout.keep_mask[i]]
    pairs += [(n+i, int(layout.next_position)+i) for i in range(generated)]
    if window >= 0:
        query_position = int(layout.next_position) + generated - 1
        pairs = [(i, p) for i, p in pairs if query_position <= p + window]
        pairs = pairs[-(window+1):]
    return pairs


def audit_trajectory_files(paths, output):
    """Rendered-token LCP versus operation-key LCP for consecutive real turns.

    This supplements, and does not replace, the server's resident-tree audit.
    The last sampled output has no committed KV and is excluded from the source.
    """
    import gzip
    import re
    from array import array
    from types import SimpleNamespace

    from sglang.srt.context_system.ir import ContextKeyData
    from sglang.srt.context_system.planner import compile_chat_program
    from sglang.srt.context_system.rules import parse_drop_rule
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.parser.gpt_oss_encoding import HarmonyEncoder

    previous, result = {}, []
    for path in sorted(map(Path, paths)):
        match = re.search(r"-(\d+)-turn-(\d+)\.json\.gz$", path.name)
        if match is None:
            continue
        case, turn = match.groups()
        with gzip.open(path, "rt") as source:
            saved = json.load(source)
        payload, response = saved["request"], saved["response"]
        messages = payload["messages"]
        clean = [{k: v for k, v in m.items() if k != "index"} for m in messages]
        ids, owners, _ = HarmonyEncoder(payload.get("reasoning_effort", "low")).render_tokens(
            clean, payload.get("tools"))
        rule = parse_drop_rule(None, messages, legacy_drop_message=payload.get("drop_message"))
        program = compile_chat_program(messages, SimpleNamespace(input_ids=ids, owners=owners),
                                       rule, payload.get("reposition"), payload.get("insert"))
        tokens = program.input_ids
        if tokens != response["sglext"]["input_ids"]:
            raise ValueError(f"Offline rendering differs from the actual served input: {path}")
        data = ContextKeyData.from_layout(program.layout)
        key = RadixKey(array("i", tokens), context=data)
        if case in previous:
            old_tokens, old_key, old_input = previous[case]
            text_lcp = common_units(old_tokens, tokens)
            exact = common_units(token_units(old_key), token_units(key))
            retry = common_units(token_units(old_key, retry=True), token_units(key, retry=True))
            result.append(dict(case=case, turn=int(turn), artifact=str(path),
                               token_lcp=text_lcp, exact=exact, retry=retry,
                               operation_extra_loss=max(0, text_lcp-retry),
                               historical_token_prefix_loss=max(0, old_input-text_lcp)))
        generated = response["sglext"]["output_ids"][0][:-1]
        data.append_tokens(array("i", generated), next_position=program.layout.next_position)
        cached = tokens + generated
        previous[case] = (cached, RadixKey(array("i", cached), context=data), len(tokens))
    report = dict(transitions=len(result),
                  operation_extra_loss=sum(r["operation_extra_loss"] for r in result),
                  token_prefix_changes=sum(r["historical_token_prefix_loss"]>0 for r in result),
                  retrigger_extensions=sum(r["retry"]>r["exact"] for r in result),
                  records=result)
    Path(output).write_text(json.dumps(report, indent=2))
    return report


def replay(args):
    """Frozen warm/exact/cold and multi-step teacher forcing on an isolated server."""
    import copy
    import gzip
    import sys
    import time

    import requests
    import torch
    from insert_summary48k import (
        digest,
        logit_metrics,
        native_logit_bounds,
        validate_completion,
    )

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "test/registered/context_system")
    )
    from serving_logits_probe import serialized_probe

    def load(path):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as source:
            return json.load(source)["request"]

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    payload = load(args.target)
    source = load(args.source) if args.source else None
    session = requests.Session()
    session.trust_env = False
    journal = []

    def flush():
        session.post(args.url + "/flush_cache", timeout=60).raise_for_status()

    def call(request, label, forced=None, probe=True):
        request = copy.deepcopy(request)
        request.update(
            model=args.model,
            n=1,
            temperature=0,
            max_tokens=args.steps if probe else 1,
            ignore_eos=True,
            return_meta_info=True,
            return_input_ids_in_sglext=True,
            return_output_ids_in_sglext=True,
        )
        request.pop("custom_logit_processor", None)
        request.pop("custom_params", None)
        path = str(output / f"{label}.pt")
        if probe:
            request["custom_logit_processor"] = serialized_probe()
            request["custom_params"] = {
                "context_trace_path": path,
                "context_trace_count": args.steps,
                "context_layer_probe": args.layer_probe,
            }
            if forced is not None:
                request["custom_params"]["context_forced_tokens"] = forced
        started = time.monotonic()
        response = session.post(
            args.url + "/v1/chat/completions", json=request, timeout=args.timeout
        )
        record = {
            "request": request,
            "response": response.json(),
            "status": response.status_code,
            "seconds": time.monotonic() - started,
        }
        with gzip.open(output / f"{label}.json.gz", "wt") as target:
            json.dump(record, target)
        response.raise_for_status()
        validate_completion(record["response"])
        journal.append(
            {
                "label": label,
                "seconds": record["seconds"],
                "usage": record["response"].get("usage"),
            }
        )
        return record["response"], (
            torch.load(path, weights_only=True) if probe else None
        )

    # Independent run: never flush the no-flush trajectory server while it runs.
    flush()
    cold_body, cold = call(payload, "cold")
    forced = cold_body["sglext"]["output_ids"][0][: args.steps]
    if len(forced) != args.steps:
        raise ValueError(
            "Frozen baseline did not generate the requested number of steps"
        )
    flush()
    if source is not None:
        call(source, "source", probe=False)
    warm_body, warm = call(payload, "warm", forced)
    exact_body, exact = call(payload, "exact", forced)
    input_hashes = [
        digest(body["sglext"]["input_ids"])
        for body in (cold_body, warm_body, exact_body)
    ]
    if len(set(input_hashes)) != 1:
        raise ValueError("Frozen replay changed execution input tokens")
    native = {
        k: v
        for k, v in payload.items()
        if k not in ("messages", "insert", "drop_message", "reposition")
    }
    native["messages"] = active_messages(payload)
    flush()
    native_cold_body, native_cold = call(native, "native-cold")
    native_forced = native_cold_body["sglext"]["output_ids"][0][: args.steps]
    if len(native_forced) != args.steps:
        raise ValueError("Native calibration generated too few forced tokens")
    native_exact_body, native_exact = call(native, "native-exact", native_forced)
    flush()
    call(dict(native, messages=native["messages"][:-1]), "native-source", probe=False)
    native_warm_body, native_warm = call(native, "native-warm", native_forced)
    native_hashes = [
        digest(body["sglext"]["input_ids"])
        for body in (native_cold_body, native_warm_body, native_exact_body)
    ]
    if len(set(native_hashes)) != 1:
        raise ValueError("Native calibration changed input tokens")
    bounds = native_logit_bounds(
        [logit_metrics(value, native_cold) for value in (native_warm, native_exact)]
    )
    metrics = {
        name: logit_metrics(value, cold)
        for name, value in (("warm", warm), ("exact", exact))
    }
    # Preserve each step's gate: aggregating the vocabulary over all eight
    # steps could hide a mean/p99 failure confined to one decode position.
    step_results = []
    for step in range(args.steps):
        step_bounds = native_logit_bounds(
            [
                logit_metrics(value[step], native_cold[step])
                for value in (native_warm, native_exact)
            ]
        )
        step_metrics = {
            name: logit_metrics(value[step], cold[step])
            for name, value in (("warm", warm), ("exact", exact))
        }
        step_results.append(
            {
                "step": step,
                "bounds": step_bounds,
                "metrics": step_metrics,
                "pass": all(
                    values[k] <= step_bounds[k]
                    for values in step_metrics.values()
                    for k in step_bounds
                ),
            }
        )
    report = {
        "target": str(args.target),
        "source": args.source,
        "input_sha256": input_hashes[0],
        "steps": args.steps,
        "bounds": bounds,
        "metrics": metrics,
        "native_input_sha256": native_hashes[0],
        "step_results": step_results,
        "requests": journal,
        "pass": all(item["pass"] for item in step_results) and all(
            values[k] <= bounds[k] for values in metrics.values() for k in bounds
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def timing(args):
    """Uninstrumented HTTP timings; run against isolated baseline/current servers.

    Each round has twelve cached ordinary and Drop/R 128-token decodes, three
    cold prefills per type, and one frozen 48K summary warm prefill. JIT/source
    population are excluded. Combine three interleaved rounds on the same GPUs.
    """
    import copy
    import gzip
    import time

    import requests
    from insert_summary48k import digest, validate_completion

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False
    records = []

    def load(path):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as source:
            return json.load(source)["request"]

    def flush():
        session.post(args.url+"/flush_cache", timeout=60).raise_for_status()

    def call(payload, label, tokens, stream=False):
        payload = copy.deepcopy(payload)
        for field in ("custom_logit_processor", "custom_params", "return_meta_info",
                      "return_input_ids_in_sglext", "return_output_ids_in_sglext"):
            payload.pop(field, None)
        payload.update(model=args.model, n=1, temperature=0, max_tokens=tokens,
                       ignore_eos=True, stream=stream)
        if stream:
            payload["stream_options"] = {"include_usage": True}
        started = time.perf_counter()
        response = session.post(args.url+"/v1/chat/completions", json=payload,
                                timeout=args.timeout, stream=stream)
        response.raise_for_status()
        first, usage = None, None
        if stream:
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                body = json.loads(line[6:])
                if body.get("usage"):
                    usage = body["usage"]
                for choice in body.get("choices", []):
                    delta = choice.get("delta", {})
                    if first is None and any(delta.get(k) for k in
                                              ("content", "reasoning_content", "tool_calls")):
                        first = time.perf_counter()
                    if choice.get("finish_reason") in ("abort", "error"):
                        raise ValueError(body)
        else:
            body = response.json()
            validate_completion(body)
            usage = body["usage"]
        ended = time.perf_counter()
        if not usage or usage["completion_tokens"] != tokens:
            raise ValueError(f"Timing sample did not complete {tokens} tokens: {usage}")
        row = dict(label=label, seconds=ended-started, usage=usage,
                   payload_sha256=digest(payload))
        if stream:
            if first is None:
                raise ValueError("No streamed token arrived")
            row.update(ttft_seconds=first-started,
                       decode_seconds_per_token=(ended-first)/(tokens-1))
        records.append(row)
        with (output/"samples.jsonl").open("a") as target:
            target.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)

    ordinary = {"messages": [
        {"role": "system", "content": "Read the reference and continue with a detailed explanation."},
        {"role": "user", "content": ("The archive lists dates, places, names, and supporting evidence. "*512)
         + "Explain how to compare and organize the evidence."}]}
    drop = {"messages": [ordinary["messages"][0],
                         {"role": "user", "content": "Discard this earlier reference. "*256},
                         {"role": "assistant", "content": "Acknowledged."},
                         ordinary["messages"][1]],
            "drop_message": {"2": [1]}, "reposition": [2]}
    for name, request in (("ordinary", ordinary), ("drop-r", drop)):
        flush()
        call(request, f"{name}-warmup", 128, True)
        for i in range(12):
            call(request, f"{name}-decode-{i}", 128, True)
        for i in range(3):
            flush()
            call(request, f"{name}-cold-{i}", 128, True)
    source, target = load(args.source), load(args.target)
    # Compile the Context shape once before the measured, independently warmed
    # request. Flush makes the measured source/target relationship reproducible.
    flush()
    call(source, "summary-source-warmup", 1)
    call(target, "summary-target-warmup", 1)
    flush()
    call(source, "summary-source", 1)
    call(target, "summary-warm-prefill", 1)
    (output/"report.json").write_text(json.dumps(records, indent=2))
    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", required=True, help="Isolated server: this replay flushes its cache"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--target", required=True, help="Saved request/response .json[.gz]"
    )
    parser.add_argument("--source", help="Prior request to populate a warm source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--layer-probe", action="store_true")
    parser.add_argument("--timing", action="store_true", help="No probes; isolated performance round")
    arguments = parser.parse_args()
    if not 1 <= arguments.steps <= 32:
        parser.error("--steps must be between 1 and 32")
    if arguments.timing:
        if not arguments.source or arguments.layer_probe:
            parser.error("Timing requires --source and forbids --layer-probe")
        timing(arguments)
    else:
        replay(arguments)
