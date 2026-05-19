"""Example entrypoint showing how to add custom training behavior without editing core loops.

Usage:
  python train_custom.py --dataset shakespeare_char --compile=False

This file demonstrates two extension patterns:
1) New training file: subclass ``Trainer`` and override setup hooks.
2) In-place in ``train.py``: copy the same logic into ``Trainer.__init__`` after
   ``self.setup()`` and keep the core train loop unchanged.
"""

import torch

from train import Trainer
from train_args import parse_args


class CustomTrainer(Trainer):
    """Trainer with an auxiliary z-loss term added to the base loss."""

    def __init__(self, args, model_group, training_group, logging_group):
        # Configure custom knobs before super() so they're available everywhere.
        self.z_loss_weight = float(getattr(args, "z_loss_weight", 1e-4))
        super().__init__(args, model_group, training_group, logging_group)

        # Keep the original loss object and wrap it with custom logic.
        base_loss_fn = self.loss_fn

        def custom_loss(logits, targets, *loss_args, **loss_kwargs):
            """Base CE loss + small z-loss regularizer on log-sum-exp(logits)."""
            base = base_loss_fn(logits, targets, *loss_args, **loss_kwargs)
            z = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
            return base + self.z_loss_weight * z

        # Optional: preserve helper methods used by existing logging code.
        if hasattr(base_loss_fn, "set_model"):
            custom_loss.set_model = base_loss_fn.set_model
        if hasattr(base_loss_fn, "bit_statistics"):
            custom_loss.bit_statistics = base_loss_fn.bit_statistics

        self.loss_fn = custom_loss


def main():
    args, model_group, training_group, logging_group = parse_args()

    # Allow CLI override if parse_args supports unknown attributes.
    if not hasattr(args, "z_loss_weight"):
        args.z_loss_weight = 1e-4

    trainer = CustomTrainer(args, model_group, training_group, logging_group)

    if not args.sample_only:
        trainer.train()


if __name__ == "__main__":
    main()
