import unittest

import torch

from BiAttn.model import BiAttnConfig, BiAttnLM
from BiAttn.left_v2 import (
    BinaryPathAttention,
    PATH_TOPOLOGIES,
    binary_path_reachability,
    build_binary_path_indices,
    path_multiplicity,
    sbda_stage,
    topology_offsets,
    SBDAttention,
)


class BiAttnTest(unittest.TestCase):
    def test_sbda_custom_paths_cannot_read_future(self) -> None:
        torch.manual_seed(42)
        for use_sdpa in (False, True):
            attn = SBDAttention(32, 4, 3, kv_heads=2, use_fused=False,
                               use_sdpa=use_sdpa)
            x = torch.randn(1, 8, 32, requires_grad=True)
            paths = torch.arange(8).view(1, 8, 1).expand(1, 8, 2).clone()
            paths[:, :, 1] = 7
            y = attn(x, path_indices=paths)
            changed = x.detach().clone()
            changed[:, 4:] += 10
            torch.testing.assert_close(y[:, :4], attn(changed, path_indices=paths)[:, :4])
            y[:, :4].square().sum().backward()
            self.assertEqual(x.grad[:, 4:].abs().max().item(), 0)

    def test_sbda_empty_source_rows_have_zero_output_and_gradient(self) -> None:
        for use_sdpa in (False, True):
            attn = SBDAttention(32, 4, 3, use_fused=False, use_sdpa=use_sdpa)
            x = torch.randn(1, 8, 32, requires_grad=True)
            paths = torch.zeros(1, 8, 2, dtype=torch.long)
            y = attn(x, path_indices=paths, path_valid=torch.zeros_like(paths, dtype=torch.bool))
            self.assertEqual(y.abs().max().item(), 0)
            y.sum().backward()
            self.assertEqual(x.grad.abs().max().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
                         "BF16 Triton validation requires Ampere or newer")
    def test_sbda_fused_gqa_bf16_matches_fp32_reference(self) -> None:
        from BiAttn.sbda_triton import fused_sbda_gqa_key_backward
        # Call the kernel directly: silent eager fallback must not pass this test.
        for length, offsets in ((1, (0, 1)), (7, (0, 1, 2, 1024)), (129, (0, 1, 2, 4))):
            torch.manual_seed(42)
            q = torch.randn(1, 4, length, 40, device="cuda", dtype=torch.bfloat16).requires_grad_()
            k = torch.randn(1, 2, length, 40, device="cuda", dtype=torch.bfloat16).requires_grad_()
            v = torch.randn_like(k).requires_grad_()
            off = torch.tensor(offsets, device="cuda", dtype=torch.int32)
            actual = fused_sbda_gqa_key_backward(q, k, v, off)
            rq, rk, rv = [t.detach().float().requires_grad_() for t in (q, k, v)]
            ix = torch.arange(length, device="cuda")[:, None] - off[None, :]
            keys = rk.repeat_interleave(2, 1)[:, :, ix.clamp_min(0).long(), :]
            vals = rv.repeat_interleave(2, 1)[:, :, ix.clamp_min(0).long(), :]
            scores = (rq.unsqueeze(-2) * keys).sum(-1) * (40 ** -0.5)
            weights = scores.masked_fill(ix[None, None] < 0, float("-inf")).softmax(-1)
            expected = (weights.unsqueeze(-1) * vals).sum(-2)
            torch.testing.assert_close(actual.float(), expected, atol=0.015, rtol=0.015)
            grad = torch.randn_like(actual)
            actual.backward(grad)
            expected.backward(grad.float())
            for original, reference in zip((q, k, v), (rq, rk, rv)):
                torch.testing.assert_close(original.grad.float(), reference.grad, atol=0.03, rtol=0.03)



    def test_forward_and_loss(self) -> None:
        model = BiAttnLM(BiAttnConfig(vocab_size=32, dim=32, layers=2, heads=4))
        inputs = torch.tensor([[1, 0, 0, 0], [1, 2, 0, 0], [1, 2, 3, 0]])
        valid = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        targets = torch.tensor([2, 3, 4])
        loss = model.triangle_loss(inputs, valid, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.next_logits(inputs, valid).shape, (3, 32))

    def test_left_v2_binary_paths_cover_the_causal_prefix(self) -> None:
        reachable = binary_path_reachability(32, 5)[-1]
        self.assertEqual(reachable, set(range(32)))
        indices = build_binary_path_indices(8, layer=3)
        self.assertEqual(indices.tolist(), [
            [0, 0], [1, 0], [2, 0], [3, 0],
            [4, 0], [5, 1], [6, 2], [7, 3],
        ])

    def test_left_v2_forward_is_causal_and_differentiable(self) -> None:
        torch.manual_seed(31)
        model = BiAttnLM(BiAttnConfig(
            vocab_size=32, dim=32, layers=5, heads=4,
            attention_backend="left_v2",
        ))
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        changed = inputs.clone()
        changed[:, 5:] = torch.tensor([[27, 28]])
        with torch.no_grad():
            first = model.forward_hidden(inputs)
            second = model.forward_hidden(changed)
        torch.testing.assert_close(first[:, :5], second[:, :5])
        loss = model(inputs).square().mean()
        loss.backward()
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters() if parameter.requires_grad
        ))

    def test_sbda_is_a_first_class_backend(self) -> None:
        self.assertEqual(sbda_stage(1), "binary")
        self.assertEqual(sbda_stage(2), "binary")
        self.assertEqual(sbda_stage(3), "local_diffusion")
        model = BiAttnLM(BiAttnConfig(
            vocab_size=32, dim=32, layers=4, heads=4,
            attention_backend="sbda",
        ))
        self.assertIsInstance(model.layers[0].attn, SBDAttention)
        self.assertEqual(model.layers[0].attn.topology, "sbda")
        output = model(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))
        self.assertEqual(output.shape, (1, 8, 32))
        self.assertTrue(torch.isfinite(output).all())

    def test_sbda_schedule_is_causal_and_staged(self) -> None:
        for layer in range(1, 7):
            offsets = topology_offsets(
                layer, topology="sbda", total_layers=6,
            )
            self.assertEqual(offsets[0], 0)
            self.assertTrue(all(offset >= 0 for offset in offsets))
            if layer <= 2:
                self.assertEqual(len(offsets), 2)
            else:
                self.assertEqual(len(offsets), 4)

    def test_all_sparse_topologies_forward(self) -> None:
        inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        for topology in PATH_TOPOLOGIES:
            torch.manual_seed(37)
            model = BiAttnLM(BiAttnConfig(
                vocab_size=32, dim=32, layers=4, heads=4,
                attention_backend="left_v2", path_topology=topology,
            ))
            output = model(inputs)
            self.assertEqual(output.shape, (1, 8, 32))
            self.assertTrue(torch.isfinite(output).all())

    def test_left_v2_sdpa_matches_manual_route(self) -> None:
        torch.manual_seed(43)
        sdpa = BinaryPathAttention(
            32, 4, layer=3, topology="binary", use_sdpa=True,
        )
        manual = BinaryPathAttention(
            32, 4, layer=3, topology="binary", use_sdpa=False,
        )
        manual.load_state_dict(sdpa.state_dict())
        x = torch.randn(2, 17, 32)
        valid = torch.ones(2, 17, dtype=torch.bool)
        valid[1, 15:] = False
        seam = torch.zeros(2, 17, dtype=torch.bool)
        seam[:, 0] = True
        seam[1, 9] = True
        with torch.no_grad():
            expected = manual(x, valid, seam)
            actual = sdpa(x, valid, seam)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_binary_and_radix4_have_one_canonical_path(self) -> None:
        for topology in ("binary", "radix4", "radix8", "mixed_local_radix4"):
            multiplicity = path_multiplicity(256, 8, topology=topology)
            self.assertEqual(max(multiplicity), 1)
            self.assertTrue(all(value == 1 for value in multiplicity))

    def test_dyadic_schedule(self) -> None:
        config = BiAttnConfig(vocab_size=32, layers=8)
        config.validate()
        self.assertEqual(config.window_schedule(), (2, 4, 8, 16, 32, 64, 128, 256))

    def test_deep_schedule_is_derived_from_layers(self) -> None:
        config = BiAttnConfig(vocab_size=32, layers=16)
        config.validate()
        self.assertEqual(
            config.window_schedule(),
            (2, 4, 8, 16, 32, 64, 128, 256,
             512, 1024, 2048, 4096, 8192, 16384, 32768, 65536),
        )

    def test_sixteen_layers_accept_dynamic_short_chapters(self) -> None:
        torch.manual_seed(23)
        model = BiAttnLM(BiAttnConfig(
            vocab_size=64, dim=16, layers=16, heads=4,
            attention_backend="bidirectional", causal_right=True,
        ))
        model.eval()
        for length in (1, 3, 7, 31, 1024):
            inputs = torch.randint(0, 64, (1, length))
            with torch.no_grad():
                hidden = model.forward_hidden(inputs)
            self.assertEqual(hidden.shape, (1, length, 16))
            self.assertTrue(torch.isfinite(hidden).all())


if __name__ == "__main__":
    unittest.main()
