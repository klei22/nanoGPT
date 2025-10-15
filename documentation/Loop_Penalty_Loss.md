# Loop Penalty Loss

The `loop_penalty` loss extends the standard cross-entropy objective with an
additional term that discourages the model from falling into short repeated
character loops (such as consecutive newlines or repeated punctuation). The
penalty is lightweight and only relies on the logits/targets tensors that are
already produced during training.

## How it works

For every position in the batch the loss looks back over the previous
`loop_penalty_window` tokens and measures how much probability mass the model
assigns to re-emitting those tokens. The mean probability of reusing one of the
recent tokens becomes a penalty term which is added to the base
cross-entropy loss. When the newline token id is provided the loss multiplies
penalties that stem from repeating the newline token, which targets the common
"\n\n\n" failure mode observed when sampling from autoregressive language
models.

Mathematically the loss is:

```
loss = cross_entropy(logits, targets)
       + loop_penalty_strength * mean(prob_repeat(tokens <= window))
```

where `prob_repeat` is the total predicted probability of generating any of the
recent tokens.

## Enabling the loss

```bash
python train.py \
    --loss_fn loop_penalty \
    --loop_penalty_strength 0.2 \
    --loop_penalty_window 4 \
    --loop_penalty_newline_id 198  # example id for "\n" in GPT-2 BPE
```

Arguments:

* `--loop_penalty_strength` – scales the additional penalty term. Start around
  `0.1` and increase if the model still falls into loops.
* `--loop_penalty_window` – how many previous tokens should be considered. A
  value of `3–6` usually captures both immediate repeats and short loops.
* `--loop_penalty_newline_id` – optional token id to treat as the newline
  character. When omitted the loss still discourages general token repeats.
* `--loop_penalty_newline_multiplier` – weight applied to repeated newline
  tokens (defaults to `2.0`). Increase if newline loops remain problematic.

The loss is compatible with multi-context or scheduled training. It can be
added to a schedule via `--loss_schedule` just like the other entries in
`LOSS_VARIANTS`.
