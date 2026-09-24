import unittest
from analyze_pd_e2e_breakdown import partition, critical_path, contribution, request_ledger
from profile_pd_e2e_breakdown import identity, bind_stats_identity


class Accounting(unittest.TestCase):
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
