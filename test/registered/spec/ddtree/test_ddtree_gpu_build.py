import unittest

import torch

from sglang.srt.speculative.ddtree_utils import (
    build_child_maps_from_parent_metadata,
    build_ddtree_tree,
    build_ddtree_tree_gpu,
)


def _heap_cap(budget: int) -> int:
    return 1 << (max(1, 2 * budget + 3) - 1).bit_length()


def _make_inputs(bs: int, depth_limit: int, budget: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20260707 + budget)
    # Use scaled random scores so this test validates builder mechanics instead of
    # near-tie floating-point ordering. Real topK values are already sorted by rank.
    raw = torch.randn((bs, depth_limit, budget), generator=gen, device="cuda") * 10.0
    vals = torch.sort(raw, dim=-1, descending=True).values
    vals = vals - torch.arange(budget, device="cuda", dtype=torch.float32) * 1.0e-3
    ids = (
        torch.arange(budget, device="cuda", dtype=torch.long).view(1, 1, budget)
        + torch.arange(depth_limit, device="cuda", dtype=torch.long).view(
            1, depth_limit, 1
        )
        * 1000
        + torch.arange(bs, device="cuda", dtype=torch.long).view(bs, 1, 1)
        * 100000
    )
    return vals, ids


@unittest.skipUnless(torch.cuda.is_available(), "DDTree GPU builder tests require CUDA")
class TestDDTreeGpuBuild(unittest.TestCase):
    def test_gpu_builder_matches_cpu_heap_no_prune(self):
        for budget in (16, 32, 64, 128):
            with self.subTest(budget=budget):
                bs = 3
                depth_limit = 16
                device = torch.device("cuda")
                top_log_probs, top_token_ids = _make_inputs(bs, depth_limit, budget)

                cpu_outputs = build_ddtree_tree(
                    draft_logits=None,
                    draft_top_log_probs=top_log_probs,
                    draft_top_token_ids=top_token_ids,
                    tree_budget=budget,
                    device=device,
                    prune_to_deepest_chains=False,
                )
                (
                    cpu_node_token_ids,
                    cpu_node_depths,
                    cpu_parents,
                    cpu_child_maps,
                    cpu_visibility,
                    cpu_actual_tree_sizes,
                    cpu_actual_tree_sizes_list,
                ) = cpu_outputs

                max_nodes = budget + 1
                out_node_token_ids = torch.empty(
                    (bs, budget), dtype=torch.long, device=device
                )
                out_node_depths = torch.empty(
                    (bs, budget), dtype=torch.long, device=device
                )
                out_parents = torch.empty(
                    (bs, max_nodes), dtype=torch.long, device=device
                )
                out_visibility = torch.empty(
                    (bs, max_nodes, max_nodes), dtype=torch.bool, device=device
                )
                out_actual_tree_sizes = torch.empty(
                    (bs,), dtype=torch.long, device=device
                )
                heap_cap = _heap_cap(budget)
                heap_scores = torch.empty(
                    (bs, heap_cap), dtype=torch.float64, device=device
                )
                heap_parents = torch.empty(
                    (bs, heap_cap), dtype=torch.int32, device=device
                )
                heap_depths = torch.empty(
                    (bs, heap_cap), dtype=torch.int32, device=device
                )
                heap_ranks = torch.empty(
                    (bs, heap_cap), dtype=torch.int32, device=device
                )

                gpu_outputs = build_ddtree_tree_gpu(
                    draft_top_log_probs=top_log_probs,
                    draft_top_token_ids=top_token_ids,
                    tree_budget=budget,
                    device=device,
                    _out_node_token_ids=out_node_token_ids,
                    _out_node_depths=out_node_depths,
                    _out_parents=out_parents,
                    _out_visibility=out_visibility,
                    _out_actual_tree_sizes=out_actual_tree_sizes,
                    _heap_scores=heap_scores,
                    _heap_parents=heap_parents,
                    _heap_depths=heap_depths,
                    _heap_ranks=heap_ranks,
                )
                (
                    gpu_node_token_ids,
                    gpu_node_depths,
                    gpu_parents,
                    gpu_child_maps,
                    gpu_visibility,
                    gpu_actual_tree_sizes,
                    gpu_actual_tree_sizes_list,
                ) = gpu_outputs

                torch.testing.assert_close(gpu_node_token_ids, cpu_node_token_ids)
                torch.testing.assert_close(gpu_node_depths, cpu_node_depths)
                torch.testing.assert_close(gpu_parents, cpu_parents)
                torch.testing.assert_close(gpu_visibility, cpu_visibility)
                torch.testing.assert_close(gpu_actual_tree_sizes, cpu_actual_tree_sizes)
                self.assertEqual(gpu_actual_tree_sizes_list, cpu_actual_tree_sizes_list)
                self.assertEqual(gpu_child_maps, [])

                draft_tokens_cpu = torch.cat(
                    [
                        torch.zeros((bs, 1), dtype=torch.long, device=device),
                        gpu_node_token_ids,
                    ],
                    dim=1,
                ).cpu().tolist()
                rebuilt_child_maps = build_child_maps_from_parent_metadata(
                    draft_tokens_cpu,
                    gpu_parents.cpu().tolist(),
                    gpu_actual_tree_sizes.cpu().tolist(),
                )
                self.assertEqual(rebuilt_child_maps, cpu_child_maps)


if __name__ == "__main__":
    unittest.main()
