"""Native HTTP cancellation, uneven stopping, logprobs and KV reuse checks."""

import concurrent.futures
import json
import math
import time
import urllib.error

from profile_workload import http


def verify(url, rows):
    def generate(index, limit, *, rid=None, long=False):
        with http(url, "/generate", {
            "rid": rid or f"lifecycle-{index}-{limit}",
            "text": rows[index]["problem"] * (4 if long else 1),
            "sampling_params": {"temperature": 0, "max_new_tokens": limit},
            "return_logprob": True, "logprob_start_len": -1, "top_logprobs_num": 2,
        }, timeout=120) as response:
            return json.load(response)

    def flush():
        with http(url, "/flush_cache?timeout=15", {}, timeout=30) as response:
            assert response.status == 200

    results = []
    for repeat in range(2):
        limits = [1, 2, 5, 16, 3, 8, 12, 24]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = []
            for i, limit in enumerate(limits):
                futures.append(pool.submit(generate, i, limit, rid=f"uneven-{repeat}-{i}"))
                time.sleep(.013)
            outputs = [f.result() for f in futures]
        for output, limit in zip(outputs, limits):
            meta = output["meta_info"]
            assert meta["finish_reason"]["type"] == "length", meta
            assert meta["completion_tokens"] == limit, meta
            assert len(output["output_ids"]) == limit
            logprobs = meta["output_token_logprobs"]
            assert len(logprobs) == limit and all(math.isfinite(x[0]) for x in logprobs)
            assert len(meta["output_top_logprobs"]) == limit
        results.append({"uneven_repeat": repeat, "outputs": outputs})
        flush()

    # With a small chunk size this covers cancellation during partial prefill;
    # the later abort also exercises a pending decode continuation.
    for delay in (.03, .4):
        rid = f"cancel-{delay}"
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            doomed = pool.submit(generate, 0, 512, rid=rid, long=True)
            time.sleep(delay)
            with http(url, "/abort_request", {"rid": rid}):
                pass
            survivor = pool.submit(generate, 1, 16, rid=f"survivor-{delay}")
            try:
                aborted = doomed.result(timeout=120)
                assert aborted["meta_info"]["finish_reason"]["type"] == "abort", aborted
            except urllib.error.HTTPError as error:
                # The standard HTTP endpoint reports explicitly aborted
                # non-streaming requests as an error response.
                aborted = {"http_status": error.code, "body": error.read().decode()}
                assert error.code in (400, 500) and "abort" in aborted["body"].lower(), aborted
            survived = survivor.result(timeout=120)
            assert survived["meta_info"]["completion_tokens"] == 16
        results.append({"abort_delay_s": delay, "aborted": aborted, "survivor": survived})
        flush()
    return {"passed": True, "cases": results}
