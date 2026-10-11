import unittest

import torch

from BiAttn.model import BiAttnConfig, BiAttnLM
from BiAttn.three_track import (
    BasicLeftTrack,
    build_left_track_layout,
    build_right_track_layout,
    left_track_pool,
    right_track_pool,
)


class BasicLeftTrackTest(unittest.TestCase):
    def test_layout_is_causal_and_matches_dyadic_left_half(self) -> None:
        layout = build_left_track_layout(8, layer=3)
        self.assertEqual((layout.width, layout.half), (8, 4))
        self.assertEqual(layout.starts.tolist(), [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(layout.ends.tolist(), [0, 0, 0, 0, 0, 1, 2, 3])
        self.assertEqual(layout.valid.tolist(), [False, False, False, False, True, True, True, True])
        self.assertTrue(torch.all(layout.ends < 8))

    def test_short_layout_clamps_indices_without_shrinking_nominal_width(self) -> None:
        left = build_left_track_layout(3, layer=3)
        right = build_right_track_layout(3, layer=3)
        self.assertEqual((left.width, left.half), (8, 4))
        self.assertEqual((right.width, right.half), (8, 4))
        self.assertTrue(torch.all(left.starts < 3))
        self.assertTrue(torch.all(left.ends < 3))
        self.assertTrue(torch.all(right.starts < 3))
        self.assertTrue(torch.all(right.ends < 3))
        self.assertFalse(left.valid.any())
        self.assertFalse(right.valid.any())

    def test_prefix_pool_matches_explicit_reference(self) -> None:
        hidden = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
        layout = build_left_track_layout(8, layer=3)
        pooled = left_track_pool(hidden, layout)
        expected = torch.tensor([0, 0, 0, 0, 0, 0.5, 1.0, 1.5]).view(1, 8, 1)
        torch.testing.assert_close(pooled, expected)

    def test_gradients_and_optional_track_embedding(self) -> None:
        torch.manual_seed(7)
        hidden = torch.randn(2, 9, 16, requires_grad=True)
        module = BasicLeftTrack(16)
        pooled, layout = module(hidden, layer=2)
        self.assertEqual(pooled.shape, hidden.shape)
        loss = pooled.square().mean()
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(module.track_embedding.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_right_track_is_left_layout_mirrored_and_shifted(self) -> None:
        hidden = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
        layout = build_right_track_layout(8, layer=1)
        pooled, valid = right_track_pool(hidden, layout)
        # Width two has raw right state x[t + 1].  The one-cell right shift
        # therefore gives x[t] at destination t + 1.
        expected = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32)
        torch.testing.assert_close(pooled[0, :, 0], expected)
        self.assertEqual(valid.tolist(), [[False, True, True, True, True, True, True, True]])

    def test_right_track_masks_the_target_outside_a_triangle_row(self) -> None:
        torch.manual_seed(9)
        model = BiAttnLM(BiAttnConfig(
            vocab_size=32, dim=32, layers=3, heads=4,
            attention_backend="bidirectional",
        ))
        inputs = torch.tensor([[1, 2, 3, 4, 5, 0, 0]])
        changed = inputs.clone()
        changed[:, 5:] = torch.tensor([[27, 28]])
        valid = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        with torch.no_grad():
            first = model.forward_hidden(inputs, valid)
            second = model.forward_hidden(changed, valid)
        torch.testing.assert_close(first[:, :5], second[:, :5])

    def test_dense_causal_right_track_cannot_read_future_cells(self) -> None:
        for backend in ("left", "bidirectional"):
            torch.manual_seed(11)
            model = BiAttnLM(BiAttnConfig(
                vocab_size=32, dim=32, layers=3, heads=4,
                attention_backend=backend,
                causal_right=backend == "bidirectional",
            ))
            inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
            changed = inputs.clone()
            changed[:, 5:] = torch.tensor([[27, 28]])
            with torch.no_grad():
                first_hidden = model.forward_hidden(inputs)
                second_hidden = model.forward_hidden(changed)
                first_logits = model.lm_head(first_hidden)
                second_logits = model.lm_head(second_hidden)
            # Positions 0..4 predict tokens 1..5.  Neither their hidden
            # states nor their logits may depend on cells 5 and later.
            torch.testing.assert_close(first_hidden[:, :5], second_hidden[:, :5])
            torch.testing.assert_close(first_logits[:, :5], second_logits[:, :5])

    def test_triangle_last_valid_logit_cannot_read_target_or_padding(self) -> None:
        inputs = torch.tensor([[1, 2, 3, 4, 5, 0, 0]])
        changed = inputs.clone()
        changed[:, 5:] = torch.tensor([[27, 28]])
        valid = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        for backend in ("left", "bidirectional"):
            torch.manual_seed(19)
            model = BiAttnLM(BiAttnConfig(
                vocab_size=32, dim=32, layers=3, heads=4,
                attention_backend=backend,
            ))
            with torch.no_grad():
                first = model.next_logits(inputs, valid)
                second = model.next_logits(changed, valid)
            # Row length five predicts input token 5.  Token 5 is absent from
            # the row and must not be recoverable through either track.
            torch.testing.assert_close(first, second)

    def test_short_sequence_matches_same_prefix_in_padded_tile(self) -> None:
        torch.manual_seed(29)
        short = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        padded = torch.cat((short, torch.zeros((1, 9), dtype=torch.long)), dim=1)
        valid = torch.zeros_like(padded, dtype=torch.bool)
        valid[:, :short.shape[1]] = True
        for backend in ("left", "bidirectional"):
            torch.manual_seed(31)
            model = BiAttnLM(BiAttnConfig(
                vocab_size=32, dim=32, layers=4, heads=4,
                attention_backend=backend,
                causal_right=backend == "bidirectional",
            ))
            with torch.no_grad():
                short_hidden = model.forward_hidden(short)
                padded_hidden = model.forward_hidden(padded, valid)
            # Padding or a fixed-size accelerated tile must not change the
            # result for the real chapter prefix.
            torch.testing.assert_close(
                short_hidden, padded_hidden[:, :short.shape[1]]
            )


if __name__ == "__main__":
    unittest.main()
