# Multicontext Mode Overview

This document explains how nanoGPT's multicontext mode works across training (`train.py`), inference (`sample.py`), and the core model (`model.py`). It also highlights the current inference algorithm and practical steps to keep multicontext sampling closer in speed to single-context generation.

## What is multicontext?

Multicontext mode trains and samples a single transformer across multiple aligned contexts (e.g., different tokenizations or auxiliary signals). During each forward pass, the model embeds every context, sums their representations token-wise, and produces per-context logits. This allows the model to co-train on heterogeneous inputs while sharing the transformer stack.

Key configuration flags:

- `--training_mode multicontext` in `train.py` enables multicontext data loading and batching.
- `--multicontext` in `sample.py` switches inference into the multicontext sampler.
- `--multicontext_datasets` lists datasets/contexts used in both training and inference. The order must match between the checkpoint config and runtime arguments.
- `--multicontext_start` provides one prompt per context for sampling.
- Optional `--numerical_multicontext` replaces token embeddings with per-context MLPs for numeric inputs.

## Training pipeline (`train.py`)

1. **Dataset loading:** When `training_mode` is `multicontext`, `load_data()` memmaps each dataset and records its vocab size. The model configuration receives `multicontext=True` and an ordered list of `vocab_sizes` so embeddings and heads are created for every context.【F:train.py†L693-L724】
2. **Batching:** `get_batch()` draws the same random offsets across all contexts, returning dictionaries of input and target tensors keyed by dataset name plus the ordered dataset list.【F:train.py†L1056-L1084】
3. **Forward/backward:** Training calls `model(..., token_dict=x_dict, target_dict=y_dict, token_order=self.args.multicontext_datasets)` to ensure the token order is consistent with the configured datasets. The loss averages per-context losses by default.【F:train.py†L1049-L1066】【F:train.py†L1822-L1832】

## Model behavior (`model.py`)

- **Multicontext embedding:** The `forward` path accepts either a `token_dict` or an ordered `token_list` plus `token_order`. It builds embeddings for each context (summing them), applies shared transformer blocks, and returns a list of logits—one tensor per context. Numerical multicontext routes through per-context MLPs for both embeddings and outputs.【F:model.py†L406-L486】
- **Inference slicing:** During inference (when no targets are provided), each logit tensor is truncated to the last time step to avoid unnecessary computation, mirroring the single-context fast path.【F:model.py†L488-L519】
- **Ordering guarantees:** The new `token_order`/`token_list` arguments remove reliance on Python dict insertion order and avoid per-call dict-to-list conversions, trimming Python overhead during multicontext sampling.【F:model.py†L392-L418】

## Inference algorithm (`sample.py`)

1. **Prompt preparation:** For each dataset name in `--multicontext_datasets`, `sample.py` loads its tokenizer metadata, encodes the provided `--multicontext_start` string, and seeds per-context token tensors. The dataset order is frozen as a tuple to match the model configuration.【F:sample.py†L1332-L1377】
2. **Generation loop:** For every new token:
   - Construct `idx_cond_list`, which slices each context's token history to the model block size.
   - Call `model(..., token_list=idx_cond_list, token_order=dataset_names)` to obtain ordered logits for all contexts in a single forward pass.
   - For categorical contexts, apply temperature/top-k sampling; for numerical contexts, round and clamp predictions according to tokenizer metadata.
   - Append sampled tokens back into the per-context buffers.
   This list-based path avoids building new dictionaries every step and preserves consistent ordering between inputs and outputs.【F:sample.py†L1384-L1429】
3. **Decoding:** After `max_new_tokens`, each context decodes its token stream with the matching tokenizer and prints the result.【F:sample.py†L1431-L1440】

## Performance notes and speed tips

Multicontext sampling does more work than single-context generation because it must embed and sample every context. The following practices keep overhead low:

- **Use the list-based sampler path:** The new `token_list`/`token_order` fast path in `model.forward` eliminates repeated dict traversals during inference and training eval, reducing Python overhead per token.【F:model.py†L392-L418】【F:sample.py†L1384-L1429】
- **Align dataset ordering:** Always pass `--multicontext_datasets` in the same order used during training so embeddings and heads are accessed without extra checks or reshuffling.【F:train.py†L693-L724】【F:sample.py†L1328-L1337】
- **Keep block sizes tight:** `sample.py` already slices each context to the model block size before calling the model, matching the single-context optimization that only forwards the final position through the LM heads.【F:sample.py†L1384-L1395】【F:model.py†L500-L519】
- **Leverage existing accelerators:** Flags like `--compile`, `--use_flash_lobo`, and mixed-precision settings apply to multicontext mode as well; enabling them narrows the gap to single-context throughput.
- **Minimize Python work in the loop:** Avoid unnecessary logging inside the token loop and reuse allocated tensors when possible (e.g., by reusing `token_list` buffers as in the current sampler) to keep GPU kernels fed.

Future work to close the remaining gap could include adding KV-caching across contexts, fusing per-context sampling operations, and supporting grouped generation where contexts share a batch dimension for even fewer kernel launches.
