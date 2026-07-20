import unittest
from types import SimpleNamespace

import torch

from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ddtree_utils import compile_ddtree_tree
from sglang.srt.speculative.spec_utils import spec_need_hidden_states


class TestDDTreeMetadata(unittest.TestCase):
    def test_runtime_reserves_target_tree_width(self):
        args = SimpleNamespace(
            speculative_algorithm="DDTREE",
            speculative_num_draft_tokens=16,
            speculative_ddtree_budget=32,
            speculative_adaptive=False,
        )
        self.assertEqual(
            ServerArgs.max_speculative_num_draft_tokens.func(args),
            33,
        )

    def test_overlap_does_not_relay_hidden_states(self):
        args = SimpleNamespace(
            speculative_algorithm="DDTREE",
            enable_multi_layer_eagle=False,
        )
        self.assertFalse(spec_need_hidden_states(args))

    def test_compile_mask_is_request_packed(self):
        device = torch.device("cpu")
        root_ids = torch.tensor([10, 20], dtype=torch.long)
        node_ids = torch.tensor([[11, 12], [21, 22]], dtype=torch.long)
        node_depths = torch.tensor([[1, 2], [1, 1]], dtype=torch.long)
        visibility = torch.tensor(
            [
                [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
                [[1, 0, 0], [1, 1, 0], [1, 0, 1]],
            ],
            dtype=torch.bool,
        )
        past_lens = torch.tensor([2, 1], dtype=torch.long)
        actual_sizes = torch.tensor([3, 3], dtype=torch.long)

        input_ids, positions, packed_mask, _ = compile_ddtree_tree(
            root_token_ids=root_ids,
            node_token_ids=node_ids,
            node_depths=node_depths,
            visibility=visibility,
            start_positions=past_lens,
            past_lengths=past_lens,
            tree_budget=2,
            actual_tree_sizes=actual_sizes,
            device=device,
            past_lens_cpu=[2, 1],
            actual_sizes_cpu=[3, 3],
            verify_token_num=3,
            build_attention_mask=True,
        )

        torch.testing.assert_close(
            input_ids, torch.tensor([[10, 11, 12], [20, 21, 22]])
        )
        torch.testing.assert_close(
            positions, torch.tensor([[2, 3, 4], [1, 2, 2]])
        )

        first_numel = 3 * (2 + 3)
        first = packed_mask[:first_numel].view(3, 5)
        second = packed_mask[first_numel:].view(3, 4)
        self.assertTrue(bool(first[:, :2].all()))
        self.assertTrue(bool(second[:, :1].all()))
        torch.testing.assert_close(first[:, 2:], visibility[0])
        torch.testing.assert_close(second[:, 1:], visibility[1])


if __name__ == "__main__":
    unittest.main()
