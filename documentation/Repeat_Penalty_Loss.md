# Repeat Penalty Loss

The `repeat_penalty` loss augments the standard cross-entropy objective with a
lightweight term that discourages the model from reusing very recent tokens. By
reducing the probability mass assigned to any token that appeared in the last
few positions, the objective encourages more varied generations and avoids
falling back on local repetition.

## How it works

For every position in the batch the loss looks back over the previous
`repeat_penalty_window` tokens and measures how much probability mass the model
assigns to re-emitting any of them. The unique set of recently emitted tokens is
constructed per position so that the penalty only counts each token once even if
it occurred multiple times in the window. The mean probability of reusing one of
the recent tokens becomes a penalty term which is added to the base
cross-entropy loss.

Mathematically the loss is:

```
loss = cross_entropy(logits, targets)
       + repeat_penalty_strength * mean(prob_repeat(tokens <= window))
```

where `prob_repeat` is the total predicted probability of generating any of the
recent tokens.

## Enabling the loss

```bash
python train.py \
    --loss_fn repeat_penalty \
    --repeat_penalty_strength 0.2 \
    --repeat_penalty_window 4
```

Arguments:

* `--repeat_penalty_strength` – scales the additional penalty term. Start around
  `0.1` and increase if repeated phrases still show up when sampling.
* `--repeat_penalty_window` – how many previous tokens should be considered. A
  value of `3–6` usually captures both immediate repeats and short phrases.

The loss is compatible with multi-context or scheduled training. It can be
added to a schedule via `--loss_schedule` just like the other entries in
`LOSS_VARIANTS`. The legacy name `loop_penalty` is kept as an alias so older
configs continue to run.
