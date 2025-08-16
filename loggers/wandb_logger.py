def setup_wandb(args, master_process: bool):
    """Initialize a Weights & Biases run if enabled."""
    if getattr(args, "wandb_log", False) and master_process:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)
        args.csv_name = args.wandb_run_name

def log_validation_metrics(trainer, losses, running_mfu):
    """Log validation metrics to Weights & Biases."""
    if trainer.args.wandb_log and trainer.master_process:
        import wandb
        wandb.log({
            "iter": trainer.iter_num,
            "train/loss": losses['train'],
            "val/loss": losses['val'],
            "lr": trainer.lr,
            "mfu": running_mfu * 100,
        })

def log_train_metrics(trainer, loss_training, running_mfu):
    """Log training metrics (non-validation) to Weights & Biases."""
    if trainer.args.wandb_log and trainer.master_process:
        import wandb
        wandb.log({
            "iter": trainer.iter_num,
            "train/loss": loss_training,
            "lr": trainer.lr,
            "mfu": running_mfu * 100,
        })

def finish_wandb():
    """Finish the Weights & Biases run."""
    import wandb
    wandb.log({"finished": True})
    wandb.finish()
