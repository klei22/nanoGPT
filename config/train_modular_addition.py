# Configuration for a miniature character-level Shakespeare model

out_dir = 'out-modular-arithmetic'
eval_interval = 10
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = True
wandb_project = 'out-modular-addition'
wandb_run_name = 'modular-addition'

dataset = 'modular_addition'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 24

# Model parameters
n_layer = 4
n_head = 4
n_embd = 32
dropout = 0.2

# Training parameters
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

# Uncomment the lines below if running on a MacBook without GPU
# device = 'cpu'
# compile = False

