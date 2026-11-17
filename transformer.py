"""Batch-first encoder-decoder Transformer from Attention Is All You Need.

Token tensors have shape (batch, length). All boolean masks use True to block
attention. Attention weights are returned before dropout so unmasked rows sum
to one; fully masked rows have zero weights and a zero attention result.
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


def make_padding_mask(tokens: Tensor, pad_idx: int = 0) -> Tensor:
    """Return a boolean (batch, length) mask for padding token IDs."""
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, length)")
    return tokens.eq(pad_idx)


def make_causal_mask(length: int, device=None) -> Tensor:
    """Return a (length, length) mask blocking positions after each query."""
    if length < 0:
        raise ValueError("length must be nonnegative")
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[Tensor, Tensor]:
    """Attend over keys, returning (output, weights).

    Inputs have shape (..., query/key length, feature size). A boolean mask
    must broadcast to (..., query length, key length). Query/key feature sizes
    must match; value may have a different feature size.
    """
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("attention masks must be boolean (True blocks attention)")
        scores = scores.masked_fill(mask, float("-inf"))
        # Softmax over an entirely blocked row would otherwise produce NaNs.
        scores = scores.masked_fill(mask.all(dim=-1, keepdim=True), 0.0)
    weights = torch.softmax(scores, dim=-1)
    if mask is not None:
        weights = weights.masked_fill(mask, 0.0)
    used_weights = dropout(weights) if dropout is not None else weights
    return torch.matmul(used_weights, value), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by positive num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return (batch, query length, d_model) output and per-head weights.

        mask may be (Q, K), (batch, Q, K), or broadcastable (batch, heads, Q, K).
        key_padding_mask must be (batch, K). Both use True to block attention.
        """
        if any(x.ndim != 3 or x.size(-1) != self.d_model for x in (query, key, value)):
            raise ValueError("query, key and value must have shape (batch, length, d_model)")
        if query.size(0) != key.size(0) or key.shape[:2] != value.shape[:2]:
            raise ValueError("batch sizes and key/value lengths must match")
        batch, query_length, _ = query.shape
        if mask is not None:
            if mask.dtype != torch.bool:
                raise TypeError("attention masks must be boolean")
            if mask.ndim not in (2, 3, 4):
                raise ValueError("attention mask must have 2, 3 or 4 dimensions")
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                raise TypeError("key_padding_mask must be boolean")
            if key_padding_mask.shape != (batch, key.size(1)):
                raise ValueError("key_padding_mask must have shape (batch, key length)")
            padding = key_padding_mask[:, None, None, :]
            mask = padding if mask is None else mask | padding

        def split_heads(x: Tensor, projection: nn.Linear) -> Tensor:
            return projection(x).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attended, weights = scaled_dot_product_attention(
            split_heads(query, self.query_projection),
            split_heads(key, self.key_projection),
            split_heads(value, self.value_projection),
            mask,
            self.dropout,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(batch, query_length, self.d_model)
        return self.output_projection(attended), weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model <= 0 or max_len <= 0:
            raise ValueError("d_model and max_len must be positive")
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: d_model // 2])
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.encoding.size(1):
            raise ValueError("sequence length exceeds positional encoding max_len")
        return self.dropout(x + self.encoding[:, : x.size(1)].to(dtype=x.dtype))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, padding_mask: Optional[Tensor] = None) -> Tensor:
        attended, _ = self.self_attention(x, x, x, mask, padding_mask)
        x = self.norm1(x + self.dropout(attended))
        return self.norm2(x + self.dropout(self.feed_forward(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None, memory_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        attended, _ = self.self_attention(x, x, x, tgt_mask, tgt_padding_mask)
        x = self.norm1(x + self.dropout(attended))
        attended, _ = self.cross_attention(x, memory, memory, key_padding_mask=memory_padding_mask)
        x = self.norm2(x + self.dropout(attended))
        return self.norm3(x + self.dropout(self.feed_forward(x)))


class Encoder(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, num_layers: int = 6, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, padding_mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask, padding_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, num_layers: int = 6, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(
        self, x: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None, memory_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, tgt_padding_mask, memory_padding_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
        num_heads: int = 8, num_layers: int = 6, d_ff: int = 2048,
        dropout: float = 0.1, max_len: int = 5000,
        src_pad_idx: Optional[int] = 0, tgt_pad_idx: Optional[int] = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = Encoder(d_model, num_heads, num_layers, d_ff, dropout)
        self.decoder = Decoder(d_model, num_heads, num_layers, d_ff, dropout)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self, src: Tensor, tgt: Tensor, src_padding_mask: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Map source and decoder-input IDs to raw target-vocabulary logits.

        Padding masks (batch, length) override masks inferred from pad IDs.
        Set a pad index to None to disable automatic masking for that input.
        Causal target masking is always applied. Pass shifted target tokens
        beginning with a start token when using teacher forcing.
        """
        if src.ndim != 2 or tgt.ndim != 2 or src.size(0) != tgt.size(0):
            raise ValueError("src and tgt must have shape (same batch, length)")
        if src.size(1) == 0 or tgt.size(1) == 0:
            raise ValueError("source and target sequences must be nonempty")
        if src_padding_mask is None and self.src_pad_idx is not None:
            src_padding_mask = make_padding_mask(src, self.src_pad_idx)
        if tgt_padding_mask is None and self.tgt_pad_idx is not None:
            tgt_padding_mask = make_padding_mask(tgt, self.tgt_pad_idx)
        source = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        target = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        memory = self.encoder(source, padding_mask=src_padding_mask)
        decoded = self.decoder(
            target, memory, make_causal_mask(tgt.size(1), tgt.device),
            tgt_padding_mask, src_padding_mask,
        )
        return self.output_projection(decoded)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = Transformer(32, 40, d_model=32, num_heads=4, num_layers=2, d_ff=64, dropout=0.0)
    model.eval()
    src = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    tgt = torch.tensor([[1, 6, 7], [1, 8, 0]])
    with torch.no_grad():
        logits = model(src, tgt)
    assert logits.shape == (2, 3, 40)
    assert torch.isfinite(logits).all()
    print("Logits shape:", tuple(logits.shape))
