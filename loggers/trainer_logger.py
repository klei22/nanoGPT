import math


def log_validation_step(trainer, losses, running_mfu, current_epoch, current_dataset):
    """Log validation metrics and send them to any configured loggers."""
    if trainer.args.dataset_list is not None:
        for dataset, dataset_losses in losses['datasets'].items():
            better_than_chance = trainer.model_args['vocab_size'] / math.exp(dataset_losses['val'].item())
            log_message = f"step {trainer.iter_num}: "
            log_message += f"{dataset:<20s}"
            log_message += f", {trainer.model.num_param}"
            log_message += f", train loss {dataset_losses['train']:.4f}"
            log_message += f", train_stdev {dataset_losses['train_std']:.4f}"
            log_message += f", btc_val_set {better_than_chance:.2e}"
            log_message += f", btc_val_per_param {(better_than_chance/trainer.model.num_param):.2e}"
            log_message += f", val loss {dataset_losses['val']:.4f}"
            log_message += f", val_stdev {dataset_losses['val_std']:.4f}"
            if trainer.args.gns_type is not None:
                log_message += f", gns {trainer.gns:.2f}"
            log_message += f", lr {trainer.lr:.4f}"
            log_message += f", tokens_trained {trainer.tokens_trained_dict[dataset]:.2e}"
            trainer.console.print(log_message)
            trainer.log_metrics(
                dataset_losses,
                running_mfu,
                trainer.epochs_trained_dict[dataset],
                trainer.tokens_trained_dict[dataset],
                dataset,
                better_than_chance,
            )
    elif trainer.args.multicontext_datasets is not None:
        for dataset, dataset_losses in losses['datasets'].items():
            log_message = f"step {trainer.iter_num}: "
            log_message += f"{dataset:<20s}"
            log_message += f", train loss {dataset_losses['train']:.4f}"
            log_message += f", train_stdev {dataset_losses['train_std']:.4f}"
            log_message += f", val loss {dataset_losses['val']:.4f}"
            log_message += f", val_stdev {dataset_losses['val_std']:.4f}"
            if trainer.args.gns_type is not None:
                log_message += f", gns {trainer.gns:.2f}"
            log_message += f", lr {trainer.lr:.4f}"
            log_message += f", tokens_trained {trainer.tokens_trained:.2e}"
            trainer.console.print(log_message)
            better_than_chance = trainer.vocab_sizes[dataset] / math.exp(dataset_losses['val'].item())
            trainer.log_metrics(
                dataset_losses,
                running_mfu,
                current_epoch,
                trainer.tokens_trained,
                dataset,
                better_than_chance,
            )
    else:
        better_than_chance = trainer.model_args['vocab_size'] / math.exp(losses['val'].item())
        log_message = f"step {trainer.iter_num}:"
        log_message += f", {trainer.model.num_param}"
        log_message += f", train loss {losses['train']:.4f}"
        log_message += f", train_stdev {losses['train_std']:.4f}"
        log_message += f", btc_val {better_than_chance:.2e}"
        log_message += f", btc_val_per_param {(better_than_chance/trainer.model.num_param):.2e}"
        log_message += f", val loss {losses['val']:.4f}"
        log_message += f", val_stdev {losses['val_std']:.4f}"
        if trainer.args.gns_type is not None:
            log_message += f", gns {trainer.gns:.2f}"
        log_message += f", batch_size {trainer.args.batch_size}"
        log_message += f", lr {trainer.lr:.4f}"
        trainer.console.print(log_message)
        trainer.log_metrics(
            losses,
            running_mfu,
            current_epoch,
            trainer.tokens_trained,
            current_dataset,
            better_than_chance,
        )


def log_train_step(trainer, lossf, dt, running_mfu, current_epoch, prior_dataset, training_losses=None):
    """Log training metrics for a single iteration."""
    log_message = f"iter {trainer.iter_num}"
    log_message += f", {dt*1000:.2f} ms"
    log_message += f", {trainer.model.num_param}"

    if trainer.args.multicontext_datasets:
        for i, mc_dataset in enumerate(trainer.args.multicontext_datasets):
            trainer.mc_btc_train[mc_dataset] = trainer.vocab_sizes[mc_dataset] / math.exp(training_losses[i].item())
            log_message += f", {trainer.underscore_abbr(mc_dataset)}"
            if trainer.args.log_btc_train:
                log_message += f" btc {trainer.mc_btc_train[mc_dataset]:.4f}"
            log_message += f", {trainer.underscore_abbr(mc_dataset)}"
            log_message += f" loss {training_losses[i].item():.4f}"
    else:
        better_than_chance = trainer.model_args['vocab_size'] / math.exp(lossf)
        log_message += f", loss {lossf:.4f}"
        if trainer.args.log_btc_train:
            log_message += f", btc_train {better_than_chance:.2e}"
        if trainer.args.log_btc_per_param:
            log_message += f", btc_train_per_param {(better_than_chance/trainer.model.num_param):.2e}"

    if trainer.args.dataset_list:
        log_message += f", epoch {trainer.epochs_trained_dict[prior_dataset]:2.2f}"
        log_message += f", tokens_trained {trainer.tokens_trained_dict[prior_dataset]:.2e}"
        log_message += f", dataset: {prior_dataset}"
    else:
        log_message += f", epoch {current_epoch:6.2f}"
        log_message += f", tokens_trained {trainer.tokens_trained:.2e}"

    log_message += f", mfu {running_mfu*100:.2f}%"
    if trainer.args.gns_type is not None:
        trainer.gns = trainer.gns_ema.get_gns()
        log_message += f", gns {trainer.gns:.2f}"
    log_message += f", batch_size {trainer.args.batch_size}"
    log_message += f", lr {trainer.lr:.4f}"
    if trainer.args.log_grad_norm:
        log_message += f", grad_norm {trainer.grad_norm:2f}"
    if trainer.args.log_grad_std:
        log_message += f", grad_std {trainer.grad_std:.2f}"

    trainer.console.print(log_message)

    if not trainer.args.multicontext_datasets:
        better_than_chance = trainer.model_args['vocab_size'] / math.exp(lossf)
    
    if trainer.args.dataset_list:
        trainer.log_metrics_non_validation(
            lossf,
            running_mfu,
            trainer.epochs_trained_dict[prior_dataset],
            trainer.tokens_trained_dict[prior_dataset],
            prior_dataset,
            better_than_chance,
        )
    if trainer.args.multicontext_datasets:
        for i, mc_dataset in enumerate(trainer.args.multicontext_datasets):
            trainer.log_metrics_non_validation(
                training_losses[i].item(),
                running_mfu,
                current_epoch,
                trainer.tokens_trained,
                mc_dataset,
                trainer.mc_btc_train[mc_dataset],
            )
    else:
        trainer.log_metrics_non_validation(
            lossf,
            running_mfu,
            current_epoch,
            trainer.tokens_trained,
            prior_dataset,
            better_than_chance,
        )
