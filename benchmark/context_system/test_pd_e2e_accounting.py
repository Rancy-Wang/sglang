import unittest
import json
import tempfile
from pathlib import Path
from analyze_pd_e2e_breakdown import partition, critical_path, contribution, request_ledger, IndexedStep, IndexedRequest
from profile_pd_e2e_breakdown import identity, bind_stats_identity, batch_identity, cache_result_metadata, seed_triton_cache, mark_attention_backend, prefill_control_cases


class Accounting(unittest.TestCase):
    def control_fixture(self):
        from build_decode_breakdown_fixture import digest, PLAN
        row = dict(case_id="fixture", rid="fixture", input_ids=list(range(8)),
                   messages=[dict(role="tool",content="old"),dict(role="tool",content="retained")],
                   replay_tokens=list(range(4)),
                   drop_state=dict(active_tokens=4,drop_message={"1":[0]},reposition=[]))
        for name, key in (("input_ids","input_sha256"),("messages","messages_sha256"),("replay_tokens","replay_sha256")):
            row[key] = digest(row[name])
        return dict(plan_id=PLAN,batch_size=1,max_new_tokens=2,require_repos=False,
                    model="test",tools=[],template_kwargs={},requests=[row])

    def test_prefill_control_has_identical_inputs_and_no_generated_feedback(self):
        fixture = self.control_fixture()
        before = json.dumps(fixture,sort_keys=True)
        full = prefill_control_cases(fixture,"no_drop",repeats=2,suffix_units=3)
        drop = prefill_control_cases(fixture,"drop",repeats=2,suffix_units=3)
        self.assertEqual(before,json.dumps(fixture,sort_keys=True))
        for i,(a,b) in enumerate(zip(full[0]["phases"],drop[0]["phases"])):
            self.assertEqual(a["payload"]["messages"],b["payload"]["messages"])
            self.assertEqual(a["payload"]["messages"][-1]["content"],"retained"+" x"*(3*i))
            self.assertEqual(a["payload"]["max_tokens"],1)
            self.assertTrue(a["payload"]["ignore_eos"])
            self.assertNotIn("drop_message",a["payload"])
            self.assertEqual(b["payload"]["drop_message"],{"1":[0]})
        self.assertEqual(len({p["payload"]["rid"] for p in full[0]["phases"]}),3)

    def test_prefill_control_rejects_extended_dropped_message(self):
        fixture = self.control_fixture()
        fixture["requests"][0]["drop_state"]["drop_message"] = {"1":[1]}
        with self.assertRaisesRegex(ValueError,"remain active"):
            prefill_control_cases(fixture,"drop")

    def test_prefill_control_rejects_nonpositive_work(self):
        for repeats, units in ((0,512),(3,0)):
            with self.assertRaises(ValueError):
                prefill_control_cases(self.control_fixture(),"drop",repeats,units)

    def test_indexed_ledger_matches_exact_union_with_gaps_and_overlaps(self):
        import random
        rng = random.Random(42)
        p = dict(prefill_bootstrap_queue_entry_time=1,wait_queue_entry_time=2,
                 forward_entry_time=3,prefill_finished_time=5)
        d = dict(wait_queue_entry_time=6,forward_entry_time=7,completion_time=9)
        for overlap in (False, True):
            for _ in range(20):
                steps, expanded = [], []
                for i in range(4):
                    start = i*2.0 if not overlap else i*1.3
                    end = start+2
                    raw = [dict(start=rng.uniform(start,end),end=end,
                                category=rng.choice(["attention","moe","unknown"])) for _ in range(10)]
                    _, pieces = partition(start,end,raw)
                    role = "prefill" if i < 2 else "decode"
                    steps.append(IndexedStep(dict(start=start,end=end,role=role,pieces=pieces)))
                    expanded.extend(dict(piece,category=role+"/"+piece["category"])
                                    for piece in pieces if piece["category"] != "unobserved")
                expected = request_ledger(0,10,p,d,expanded)
                actual = request_ledger(0,10,p,d,IndexedRequest(steps),compact=True)
                for key in expected["components"].keys() | actual["components"].keys():
                    self.assertAlmostEqual(expected["components"].get(key,0),actual["components"].get(key,0))
                self.assertAlmostEqual(actual["residual_s"],0)
                self.assertEqual(actual["pieces"],[])

    def test_index_rejects_nonexclusive_and_handles_boundary_queries(self):
        step = dict(start=0,end=4,role="decode",pieces=[
            dict(start=1,end=2,category="attention"),dict(start=2,end=3,category="attention")])
        index = IndexedStep(step)
        for start,end,expected in [(-1,0,0),(1,2,1),(2,2,0),(2.5,5,0.5),(0,4,2)]:
            self.assertAlmostEqual(index.totals(start,end)["decode/attention"],expected)
        step["pieces"].append(dict(start=2,end=4,category="moe"))
        with self.assertRaises(ValueError):
            IndexedStep(step)

    def test_direct_attention_backend_markers_preserve_calls_and_errors(self):
        from types import SimpleNamespace
        events, calls = [], []
        nvtx = SimpleNamespace(range_push=lambda label: events.append(label),
                               range_pop=lambda: events.append("pop"))
        result = object()
        class Backend:
            def forward(self, *args, **kwargs):
                calls.append((args, kwargs))
                if kwargs.get("fail"):
                    raise ValueError("original failure")
                return result
        original = Backend.forward
        mark_attention_backend(Backend, nvtx, False)
        self.assertIs(Backend.forward, original)
        mark_attention_backend(Backend, nvtx, True)
        wrapped = Backend.forward
        mark_attention_backend(Backend, nvtx, True)
        self.assertIs(Backend.forward, wrapped)
        backend = Backend()
        args = (object(), object(), object(), SimpleNamespace(sliding_window_size=-1), object())
        self.assertIs(backend.forward(*args, save_kv_cache=False), result)
        self.assertEqual(calls[-1], (args, {"save_kv_cache": False}))
        self.assertEqual(events, ["component:full_attention:attention_backend", "pop"])
        kwargs = dict(q=args[0], k=args[1], v=args[2],
                      layer=SimpleNamespace(sliding_window_size=128),
                      forward_batch=args[4], fail=True)
        with self.assertRaisesRegex(ValueError, "original failure"):
            backend.forward(**kwargs)
        self.assertEqual(calls[-1], ((), kwargs))
        self.assertEqual(events[-2:], ["component:swa_attention:attention_backend", "pop"])

    def test_compiled_cache_copies_are_isolated_and_auditable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root/"source", root/"target"
            source.mkdir(); target.mkdir()
            binary = source/"kernel.cubin"
            binary.write_bytes(b"compiled")
            group = source/"__grp__kernel.json"
            group.write_text(json.dumps({"child_paths":{"binary":str(binary)}}))
            original = group.read_bytes()
            manifest = seed_triton_cache(source,target)
            self.assertEqual(json.loads((target/group.name).read_text())["child_paths"]["binary"],str((target/binary.name).resolve()))
            (target/binary.name).write_bytes(b"changed")
            self.assertEqual(binary.read_bytes(),b"compiled")
            self.assertEqual(group.read_bytes(),original)
            second = root/"second"; second.mkdir()
            self.assertEqual(seed_triton_cache(source,second)["source_manifest_sha256"],manifest["source_manifest_sha256"])
            with self.assertRaises(ValueError): seed_triton_cache(source,target)

    def test_cache_seed_rejects_external_group_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/"source"; target = root/"target"
            source.mkdir(); target.mkdir()
            outside = root/"outside"; outside.write_bytes(b"x")
            group = source/"__grp__bad.json"
            group.write_text(json.dumps({"child_paths":{"binary":str(outside)}}))
            with self.assertRaises(ValueError): seed_triton_cache(source,target)
            self.assertFalse(list(target.iterdir()))
            group.unlink(); (source/"link").symlink_to(outside)
            with self.assertRaises(ValueError): seed_triton_cache(source,target)

    def test_overlap_is_not_double_counted(self):
        totals, _ = partition(0,10,[dict(start=1,end=7,category="P"),dict(start=4,end=9,category="D")])
        self.assertEqual(totals,{"unobserved":2,"P":3,"overlap:D+P":3,"D":2})
        self.assertEqual(sum(totals.values()),10)

    def test_same_category_nested_spans_do_not_inflate_work(self):
        totals,_ = partition(0,5,[dict(start=-1,end=3,category="attention"),dict(start=1,end=4,category="attention")])
        self.assertEqual(totals,{"attention":4,"unobserved":1})

    def test_latest_dependency_not_sum_of_parallel_work(self):
        result = critical_path([
            dict(id="P",start=0,end=8,category="prefill"),
            dict(id="alloc",start=0,end=5,category="allocation"),
            dict(id="D",start=8,end=10,category="decode",parents=["P","alloc"])],"D")
        self.assertEqual(result["nodes"],["P","D"])
        self.assertEqual(sum(result["totals"].values()),10)

    def test_missing_dependency_gap_remains_unobserved(self):
        result = critical_path([dict(id="P",start=0,end=2,category="P"),
            dict(id="D",start=4,end=5,category="D",parents=["P"])],"D")
        self.assertEqual(result["totals"]["unobserved"],2)

    def test_reject_overlap_and_cycle(self):
        for nodes in ([dict(id="a",start=0,end=2,category="a",parents=["b"]),dict(id="b",start=0,end=1,category="b",parents=["a"])],
                      [dict(id="a",start=0,end=2,category="a"),dict(id="b",start=1,end=3,category="b",parents=["a"]) ]):
            with self.assertRaises(ValueError):
                critical_path(nodes,"b")

    def test_negative_contribution_is_preserved(self):
        values = {r["category"]:r for r in contribution({"attention":8,"cpu":1},{"attention":4,"cpu":2})}
        self.assertAlmostEqual(values["attention"]["contribution_percent"],400/3)
        self.assertAlmostEqual(values["cpu"]["contribution_percent"],-100/3)
        self.assertAlmostEqual(sum(v["contribution_percent"] for v in values.values()),100)

    def test_equal_total_has_no_defined_contribution_percentage(self):
        self.assertIsNone(contribution({"a":1},{"b":1})[0]["contribution_percent"])

    def test_identity_never_serializes_history(self):
        class Request:
            rid="x"
            bootstrap_room=42
            origin_input_ids=[1]*100
        self.assertEqual(identity(Request()),{"rid":"x","bootstrap_room":42})

    def test_stats_identity_does_not_require_native_tracing(self):
        from types import SimpleNamespace
        stats = SimpleNamespace()
        req = SimpleNamespace(rid="r",bootstrap_room=123,time_stats=stats)
        bind_stats_identity(req)
        self.assertEqual(stats._e2e_identity,{"rid":"r","bootstrap_room":123})

    def test_batch_and_cache_metadata_do_not_materialize_device_values(self):
        from types import SimpleNamespace
        class DeviceShapeOnly:
            def __len__(self): return 64
            def tolist(self): raise AssertionError("GPU sync forbidden")
            def item(self): raise AssertionError("GPU sync forbidden")
        req = SimpleNamespace(rid="x",bootstrap_room=7)
        self.assertEqual(batch_identity(SimpleNamespace(reqs=[req])),
                         {"requests":[{"rid":"x","bootstrap_room":7}]})
        result = SimpleNamespace(device_indices=DeviceShapeOnly(),context_retry=True,
                                 context_exact_prefix_len=80)
        self.assertEqual(cache_result_metadata(result),dict(matched_slots=64,
                         context_retry=True,context_exact_prefix_len=80))

    def test_request_ledger_keeps_queue_and_kernel_savings_distinct(self):
        p=dict(prefill_bootstrap_queue_entry_time=1,wait_queue_entry_time=2,
               forward_entry_time=3,prefill_finished_time=5)
        d=dict(wait_queue_entry_time=6,forward_entry_time=7,completion_time=9)
        result=request_ledger(0,10,p,d,[dict(start=3.5,end=4.5,category="P/attention"),
                                       dict(start=7.5,end=8.5,category="D/attention")])
        self.assertEqual(result["residual_s"],0)
        self.assertEqual(result["components"]["p_queue"],1)
        self.assertEqual(result["components"]["P/attention"],1)
        self.assertEqual(result["components"]["d_forward_wait_or_unclassified"],1)
        with self.assertRaises(ValueError):
            request_ledger(0,10,p,dict(d,wait_queue_entry_time=4),[])


if __name__ == "__main__":
    unittest.main()
