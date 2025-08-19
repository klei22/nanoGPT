import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter

def setup_tensorboard(args):
    """Initialize a TensorBoard SummaryWriter based on args.

    Returns the writer instance or None if tensorboard logging is disabled.
    Also sets a CSV-friendly run name when csv logging is enabled.
    """
    if not getattr(args, "tensorboard_log", False):
        return None

    timestamp_prefix = time.strftime("%Y%m%d-%H%M%S")
    if getattr(args, "timestamp", None):
        timestamp_prefix = args.timestamp

    if args.tensorboard_run_name is None:
        args.tensorboard_run_name = f"{timestamp_prefix}"

    run_name = args.tensorboard_run_name
    sanitized_dataset = args.dataset.replace("/", "_")
    if getattr(args, "csv_log", False):
        args.csv_name = f"{sanitized_dataset}_{run_name}"

    log_subpath = os.path.join(args.tensorboard_log_dir, run_name)
    writer = SummaryWriter(log_subpath)
    return writer

def log_validation_metrics(trainer, losses, running_mfu, epoch, tokens_trained,
                           target_dataset, val_better_than_chance):
    """Log validation metrics to TensorBoard."""
    if not trainer.args.tensorboard_log or trainer.writer is None:
        return
    writer = trainer.writer
    writer.add_scalars(
        f"{target_dataset}/loss_iters", {"val": losses['val'].item()}, trainer.iter_num
    )
    writer.add_scalars(
        f"{target_dataset}/loss_tokens", {"val": losses['val'].item()}, tokens_trained
    )

    if trainer.args.log_btc_train:
        writer.add_scalars(
            f"{target_dataset}/chance_tokens",
            {"val_chance": val_better_than_chance},
            tokens_trained,
        )
        writer.add_scalars(
            f"{target_dataset}/chance_iters",
            {"val_chance": val_better_than_chance},
            trainer.iter_num,
        )

    if trainer.args.log_btc_per_param:
        writer.add_scalars(
            f"{target_dataset}/btc_per_param_tokens",
            {"val_chance": val_better_than_chance / trainer.model.num_param},
            tokens_trained,
        )
        writer.add_scalars(
            f"{target_dataset}/btc_per_param_iters",
            {"val_chance": val_better_than_chance / trainer.model.num_param},
            trainer.iter_num,
        )

    writer.add_scalar(f"{target_dataset}/epoch", epoch, trainer.iter_num)
    writer.add_scalar(f"{target_dataset}/tokens_trained", tokens_trained, trainer.iter_num)
    writer.add_scalar(f"{target_dataset}/vram", trainer.vram_allocated, trainer.iter_num)
    writer.add_scalar(f"{target_dataset}/mfu_pct", running_mfu * 100, trainer.iter_num)
    writer.add_scalar(
        f"{target_dataset}/loss_vocab",
        trainer.model_args['vocab_size'] / torch.exp(losses['val']).item(),
        trainer.iter_num,
    )
    writer.add_scalar(f"{target_dataset}/lr_iters", trainer.lr, trainer.iter_num)
    writer.add_scalar(f"{target_dataset}/lr_tokens", trainer.lr, tokens_trained)
    writer.add_scalar(f"{target_dataset}/batch_size_iters", trainer.args.batch_size, trainer.iter_num)
    writer.add_scalar(f"{target_dataset}/batch_size_tokens", trainer.args.batch_size, tokens_trained)
    writer.add_scalar(f"{target_dataset}/std_val_iters", losses['val_std'].item(), trainer.iter_num)
    writer.add_scalar(f"{target_dataset}/std_val_tokens", losses['val_std'].item(), tokens_trained)

    if trainer.args.gns_type is not None:
        writer.add_scalar(f"{target_dataset}/gns_iters", trainer.gns, trainer.iter_num)
        writer.add_scalar(f"{target_dataset}/gns_tokens", trainer.gns, tokens_trained)

def log_train_metrics(trainer, loss_training, running_mfu, epoch, tokens_trained,
                       target_dataset, train_better_than_chance):
    """Log training metrics (non-validation) to TensorBoard."""
    if not trainer.args.tensorboard_log or trainer.writer is None:
        return
    writer = trainer.writer
    writer.add_scalars(
        f"{target_dataset}/loss_iters", {"train": loss_training}, trainer.iter_num
    )
    writer.add_scalars(
        f"{target_dataset}/loss_tokens", {"train": loss_training}, tokens_trained
    )

    if trainer.args.log_btc_train:
        writer.add_scalars(
            f"{target_dataset}/chance_tokens",
            {"train_chance": train_better_than_chance},
            tokens_trained,
        )
        writer.add_scalars(
            f"{target_dataset}/chance_iters",
            {"train_chance": train_better_than_chance},
            trainer.iter_num,
        )

    if trainer.args.log_btc_per_param:
        writer.add_scalars(
            f"{target_dataset}/btc_per_param_tokens",
            {"train_chance": train_better_than_chance / trainer.model.num_param},
            tokens_trained,
        )
        writer.add_scalars(
            f"{target_dataset}/btc_per_param_iters",
            {"train_chance": train_better_than_chance / trainer.model.num_param},
            trainer.iter_num,
        )

    writer.add_scalar(f"{target_dataset}/mfu_pct", running_mfu * 100, trainer.iter_num)
    writer.add_scalar(f"{target_dataset}/vram", trainer.vram_allocated, trainer.iter_num)
    writer.add_scalar(f"{target_dataset}/param", trainer.model.num_param, trainer.iter_num)
    writer.add_scalar(f"{target_dataset}/epoch", epoch, trainer.iter_num)
    writer.add_scalar(f"{target_dataset}/tokens_trained", tokens_trained, trainer.iter_num)
    writer.add_scalar(f"{target_dataset}/lr_iters", trainer.lr, trainer.iter_num)
    writer.add_scalar(f"{target_dataset}/lr_tokens", trainer.lr, tokens_trained)
    writer.add_scalar(f"{target_dataset}/batch_size_iter", trainer.args.batch_size, trainer.iter_num)
    writer.add_scalar(f"{target_dataset}/batch_size_tokens", trainer.args.batch_size, tokens_trained)

    if trainer.args.log_grad_norm:
        writer.add_scalar(f"{target_dataset}/grad_norm_iters", trainer.grad_norm, trainer.iter_num)
        writer.add_scalar(f"{target_dataset}/grad_norm_tokens", trainer.grad_norm, tokens_trained)

    if trainer.args.log_grad_std:
        writer.add_scalar(f"{target_dataset}/grad_std_iters", trainer.grad_std, trainer.iter_num)
        writer.add_scalar(f"{target_dataset}/grad_std_tokens", trainer.grad_std, tokens_trained)

    if trainer.args.gns_type is not None:
        writer.add_scalar(f"{target_dataset}/gns_iters", trainer.gns, trainer.iter_num)
        writer.add_scalar(f"{target_dataset}/gns_tokens", trainer.gns, tokens_trained)

def log_gamma_beta(trainer, gamma, beta, layer_num, head_num=None):
    """Log gamma and beta parameters to TensorBoard."""
    if not trainer.args.tensorboard_log or trainer.writer is None:
        return
    writer = trainer.writer
    if head_num:
        writer.add_scalars(
            "gammas",
            {"gamma_L" + str(layer_num) + "_H" + head_num: gamma}, trainer.iter_num
        )
        writer.add_scalars(
            "betas",
            {"beta_L" + str(layer_num) + "_H" + head_num: beta}, trainer.iter_num
        )
    else:
        writer.add_scalar("gamma_L" + str(layer_num), gamma, trainer.iter_num)
        writer.add_scalar("beta_L" + str(layer_num), beta, trainer.iter_num)

def close_tensorboard(writer):
    """Flush and close the TensorBoard writer."""
    writer.flush()
    writer.close()
