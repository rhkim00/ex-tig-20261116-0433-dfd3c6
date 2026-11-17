import math
import unittest

import torch

from transformer import (
    MultiHeadAttention,
    PositionalEncoding,
    Transformer,
    make_causal_mask,
    make_padding_mask,
    scaled_dot_product_attention,
)


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def small_model(self):
        return Transformer(20, 24, d_model=12, num_heads=3, num_layers=2, d_ff=24, dropout=0.0)

    def test_forward_shape_and_backward(self):
        model = self.small_model()
        src = torch.tensor([[1, 2, 0], [3, 4, 5]])
        tgt = torch.tensor([[1, 6], [1, 7]])
        logits = model(src, tgt)
        self.assertEqual(logits.shape, (2, 2, 24))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 24), torch.tensor([6, 8, 7, 9]))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_scaled_attention_matches_hand_calculation(self):
        query = torch.tensor([[[1.0, 0.0]]])
        key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        value = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])
        output, weights = scaled_dot_product_attention(query, key, value)
        probability = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
        torch.testing.assert_close(weights, torch.tensor([[[probability, 1 - probability]]]))
        torch.testing.assert_close(output, torch.tensor([[[2 * probability, 4 * (1 - probability)]]]))

    def test_multi_head_shape_and_normalization(self):
        attention = MultiHeadAttention(12, 3, dropout=0.0)
        query = torch.randn(2, 4, 12)
        memory = torch.randn(2, 5, 12)
        output, weights = attention(query, memory, memory)
        self.assertEqual(output.shape, query.shape)
        self.assertEqual(weights.shape, (2, 3, 4, 5))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3, 4))

    def test_masks_and_padding_attention(self):
        tokens = torch.tensor([[1, 0, 0], [2, 3, 0]])
        padding = make_padding_mask(tokens)
        self.assertEqual(padding.tolist(), [[False, True, True], [False, False, True]])
        causal = make_causal_mask(3)
        self.assertEqual(causal.tolist(), [[False, True, True], [False, False, True], [False, False, False]])
        attention = MultiHeadAttention(12, 3, dropout=0.0)
        x = torch.randn(2, 3, 12)
        _, weights = attention(x, x, x, causal, padding)
        self.assertTrue((weights.masked_select(padding[:, None, None, :]) == 0).all())
        self.assertTrue((weights.masked_select(causal[None, None, :, :]) == 0).all())
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3, 3))

    def test_per_batch_mask(self):
        attention = MultiHeadAttention(12, 3, dropout=0.0)
        x = torch.randn(2, 3, 12)
        mask = torch.zeros(2, 3, 3, dtype=torch.bool)
        mask[0, :, 1] = True
        mask[1, :, 2] = True
        _, weights = attention(x, x, x, mask)
        self.assertTrue((weights[0, :, :, 1] == 0).all())
        self.assertTrue((weights[1, :, :, 2] == 0).all())

    def test_fully_masked_rows_have_finite_backward(self):
        query = torch.randn(2, 3, 4, requires_grad=True)
        key = torch.randn(2, 3, 4, requires_grad=True)
        value = torch.randn(2, 3, 4, requires_grad=True)
        output, weights = scaled_dot_product_attention(query, key, value, torch.ones(3, 3, dtype=torch.bool))
        self.assertTrue((weights == 0).all())
        self.assertTrue((output == 0).all())
        output.sum().backward()
        for tensor in (query, key, value):
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_future_target_tokens_do_not_change_prefix_logits(self):
        model = self.small_model().eval()
        src = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
        tgt = torch.tensor([[1, 7, 8, 9], [1, 10, 11, 12]])
        with torch.no_grad():
            baseline = model(src, tgt)
            for position in range(tgt.size(1) - 1):
                changed = tgt.clone()
                changed[:, position + 1 :] = 13
                actual = model(src, changed)
                torch.testing.assert_close(actual[:, : position + 1], baseline[:, : position + 1])

    def test_explicit_source_padding_mask_hides_changed_tokens(self):
        model = self.small_model().eval()
        src = torch.tensor([[2, 3, 0, 0], [4, 5, 6, 0]])
        tgt = torch.tensor([[1, 7, 8], [1, 9, 10]])
        padding = make_padding_mask(src)
        changed = src.clone()
        changed[padding] = 11
        with torch.no_grad():
            baseline = model(src, tgt)
            actual = model(changed, tgt, src_padding_mask=padding)
        torch.testing.assert_close(actual, baseline)

    def test_automatic_target_padding_mask(self):
        model = self.small_model().eval()
        src = torch.tensor([[2, 3, 0]])
        tgt = torch.tensor([[1, 0, 7]])
        changed = torch.tensor([[1, 9, 7]])
        with torch.no_grad():
            baseline = model(src, tgt)
            actual = model(src, changed, tgt_padding_mask=make_padding_mask(tgt))
        torch.testing.assert_close(actual[:, [0, 2]], baseline[:, [0, 2]])

    def test_all_padding_is_finite(self):
        model = self.small_model().eval()
        logits = model(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long))
        self.assertTrue(torch.isfinite(logits).all())

    def test_sinusoidal_encoding_supports_odd_width(self):
        encoding = PositionalEncoding(5, dropout=0.0, max_len=3)
        output = encoding(torch.zeros(2, 3, 5))
        torch.testing.assert_close(output[:, 0], torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]]).expand(2, -1))
        self.assertAlmostEqual(output[0, 1, 0].item(), math.sin(1), places=6)
        self.assertAlmostEqual(output[0, 1, 3].item(), math.cos(10000 ** (-2 / 5)), places=6)
        self.assertIn("encoding", dict(encoding.named_buffers()))
        self.assertNotIn("encoding", dict(encoding.named_parameters()))
        with self.assertRaises(ValueError):
            encoding(torch.zeros(1, 4, 5))

    def test_invalid_configuration_and_mask(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(10, 3)
        with self.assertRaises(ValueError):
            Transformer(20, 24, num_layers=0)
        attention = MultiHeadAttention(12, 3)
        x = torch.randn(1, 2, 12)
        with self.assertRaises(TypeError):
            attention(x, x, x, mask=torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            attention(x, x, x, key_padding_mask=torch.zeros(1, 3, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
