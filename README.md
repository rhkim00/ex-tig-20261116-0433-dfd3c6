# Transformer

A PyTorch implementation of the encoder-decoder architecture from *Attention
Is All You Need*. It includes scaled dot-product and multi-head attention,
sinusoidal positions, ReLU feed-forward networks, and post-residual layer
normalization. The defaults are `d_model=512`, `num_heads=8`, `num_layers=6`
in each stack, `d_ff=2048`, and `dropout=0.1`.

Requires Python 3 and PyTorch (the only external dependency, declared in
`requirements.txt`). In an environment with PyTorch installed:

```python
import torch
from transformer import Transformer

model = Transformer(
    src_vocab_size=100, tgt_vocab_size=120,
    d_model=32, num_heads=4, num_layers=2, d_ff=64,
)
src = torch.tensor([[4, 5, 6, 0]])
tgt = torch.tensor([[1, 7, 8]])  # decoder inputs, starting with a start token
logits = model(src, tgt)  # (1, 3, 120), raw logits without softmax
```

All token inputs are batch first: `(batch, length)`. Token IDs must be integer
tensors accepted by `nn.Embedding`. Source and target padding IDs default to
zero and may be configured separately with `src_pad_idx` and `tgt_pad_idx`.
Use `None` to disable automatic padding detection. Explicit boolean
`src_padding_mask` and `tgt_padding_mask` arguments override detected masks.
Masks use **True to block attention**. Padding masks exclude keys, so output
logits at padded query positions still exist; exclude those positions from any
downstream loss. Sequence lengths must be between 1 and `max_len` (default 5000).

`make_padding_mask(tokens, pad_idx)` returns `(batch, length)`;
`make_causal_mask(length, device)` returns `(length, length)`. The top-level
model always applies causal masking in decoder self-attention, and applies
source padding masks in both encoder attention and decoder cross-attention.
Standalone decoder layers/stacks take an explicit `tgt_mask`.

`MultiHeadAttention` returns `(output, weights)`, where output has the query
shape and weights have shape `(batch, heads, query_length, key_length)`.
Weights are returned before dropout and sum to one for rows with allowed keys.
Fully blocked rows have zero weights; the scaled attention result is zero
before the multi-head output projection and its bias. Attention masks may have
shape `(Q, K)`, `(batch, Q, K)`, or broadcastable `(batch, heads, Q, K)`.

Call `model.eval()` for deterministic inference. For teacher forcing, pass
targets shifted right (including a start token) and compare logits with the
next-token labels. Training, tokenization, and datasets are outside this module.

Run the small forward-pass example and behavioral tests:

```sh
python3 transformer.py
python3 -m unittest discover -s tests -v
```
