"""Confidence-triggered follow-up passes for next-token training."""

import torch
import torch.nn.functional as F


def _deferred_loss(logits, _targets, iter_num=None):
    """Keep the model on its full-sequence path; loss is masked by the caller."""
    return logits.new_zeros(())


def confidence_rethinking_forward(
    model,
    inputs,
    targets,
    *,
    thinking_token_id,
    confidence_threshold=0.70,
    max_passes=3,
    iter_num=None,
    dataset_idx=None,
):
    """Run NTP, replacing uncertain prediction positions with ``<thinking>``.

    The first pass is ordinary next-token training.  Each follow-up pass only
    trains positions whose target probability was below the threshold on the
    preceding pass.  The input at those positions is replaced by the thinking
    token, so the model gets another attempt at the *same* target token.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1")

    retry_inputs = inputs
    losses = []
    retry_mask = targets.ne(-1)
    logits = None

    for pass_idx in range(max_passes):
        logits, _ = model(
            retry_inputs,
            # Supplying targets is required even though we calculate the
            # masked loss below. GPT's targets=None inference path only
            # returns logits for the final sequence position.
            targets=targets,
            iter_num=iter_num,
            dataset_idx=dataset_idx,
            loss_fn=_deferred_loss,
        )
        flat_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction="none",
        ).view_as(targets)
        active = targets.ne(-1) if pass_idx == 0 else retry_mask
        if active.any():
            losses.append(flat_loss[active].mean())

        if pass_idx + 1 == max_passes:
            break
        with torch.no_grad():
            safe_targets = targets.clamp_min(0)
            target_probability = logits.softmax(dim=-1).gather(
                -1, safe_targets.unsqueeze(-1)
            ).squeeze(-1)
            retry_mask = active & target_probability.lt(confidence_threshold)
        if not retry_mask.any():
            break
        retry_inputs = retry_inputs.masked_fill(retry_mask, thinking_token_id)

    return logits, torch.stack(losses).mean()
