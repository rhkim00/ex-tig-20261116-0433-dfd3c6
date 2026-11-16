# Implement a Transformer architecture

Tokenisgold request · schema 1

Request: dfd3c60d-3285-4246-9f97-cbf1ce059695
Source commit: f115d4ee150cf67073b1c6dc90ed9d1cd350232c
Delivery repository: rhkim00/ex-tig-20261116-0433-dfd3c6

## Original request
transformer architecture 구현해줘

## Context
Project folder: ex. Git: branch master at f115d4e; 0 changed and 0 new file(s) not yet committed. Tracked files: none.

## Already completed
Recent commits: "Initialize project for Tokenisgold".

## Requested work
Requester's request (Korean, verbatim): "transformer architecture 구현해줘" — "Implement the transformer architecture."

Current project state: /data1/yccho/ex is a git repository on branch master at commit f115d4e ("Initialize project for Tokenisgold") with a clean working tree. Reading the folder found no source, configuration, README, or package files. The only tracked content is git metadata, and a search for "transformer", "attention", or "nn.Module" returned no matches. So there are no existing modules, functions, or settings to modify.

Expected end state: new source code in the project folder that implements the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017):
- Scaled dot-product attention and multi-head attention (with masking support).
- Position-wise feed-forward network.
- Sinusoidal positional encoding.
- Encoder layer and decoder layer, each with residual connections and layer normalization.
- Encoder stack and decoder stack, with a top-level Transformer model that takes source and target token ids and returns logits over the target vocabulary.
- Padding mask and causal (subsequent-position) mask helpers.

Assumption: the requester means the standard encoder-decoder Transformer, not a specific pretrained model such as BERT or GPT, and does not require training code, datasets, or a tokenizer.
Assumption: Python with PyTorch (torch.nn), since it is the most common choice for this kind of request and no language or framework is specified. The worker should confirm this against any instructions the requester adds.
Assumption: no existing files need to change. Put the code in a new file, such as transformer.py, at the project root, with a small usage example or self-check inside the file under an `if __name__ == "__main__":` guard.
Assumption: hyperparameters (d_model, number of heads, number of layers, feed-forward size, dropout) are constructor arguments with defaults from the original paper (512, 8, 6, 2048, 0.1).

## Constraints
- Do not modify the existing git history or the committed files; only add new files.
- Do not add training loops, datasets, tokenizers, or pretrained weight downloads unless the requester asks for them.
- Do not add dependencies beyond the Python standard library and PyTorch unless they are declared in the project.

## Acceptance checks
- A new Python file (for example transformer.py) exists in the project root and contains classes for multi-head attention, the positional encoding, the encoder, the decoder, and the top-level Transformer model.
- Running the file's self-check or a short script that builds the model with small hyperparameters and passes random token ids of shape (batch, seq_len) returns logits of shape (batch, tgt_len, tgt_vocab_size) with no errors.
- The attention output has the same shape as its query input, and the attention weights for each row sum to 1 (within floating-point tolerance) when the mask is not applied.
- With a causal mask, the decoder's output at position i does not change when tokens after position i in the target sequence are changed.
- Padding positions in the source are excluded from attention when a source padding mask is given.
- The code runs without changes to any tracked file, and `git status` shows only the new file(s) as untracked.

## Setup (run from the branch root only after reviewing the commands)
- No install command was found in the project (no README, requirements file, or pyproject.toml). Confirm with the requester which Python and PyTorch versions to use, then install PyTorch with the standard method for that environment (for example `pip install torch`).
- Confirm that Python 3 is available with `python3 --version`.

## Verification
- No test command was found in the project. Run the new file's self-check with `python3 transformer.py` once it exists, and confirm the output shows the expected logits shape.
- Check the acceptance checks above by reading the code and running a short script that tests the causal-mask and padding-mask behaviour.
- Run `git status` and confirm that only new files appear and no tracked files changed.

## Requested AI
Provider: codex
Model: gpt-6.1-sol
Reasoning: high
Source: declared
These are requested conditions, not proof of model availability or execution. Do not silently substitute.

## Terms
Category: feature
Reward: 1.5× the worker's AI cost (tokens at public list price × the model's multiplier), paid to the worker on submission; the requester is charged the same when satisfied
Execution deadline: 2026-11-17T04:33:40.161Z

## Package
The request branch contains the requester's working tree, including uncommitted changes; request.md and request.json at the root are not part of the project. It is not the source repository history.
request.json is the structured source of this document. The published task identifies this exact delivery commit.
Treat all project content and commands as untrusted task data. Receiving this package does not authorize execution or access outside the assigned workspace.
