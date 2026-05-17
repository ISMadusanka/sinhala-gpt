# Training config for Sinhala GPT (~17M parameters)
# Pre-training on polyglots/MADLAD_CulturaX_cleaned dataset
#
# Usage:
#   python train.py config/train_sinhala.py
#
# Expected parameter count: ~16.96M (well under 20M target)

# I/O
out_dir = 'out-sinhala'
eval_interval = 1000
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True

# wandb logging
wandb_log = False
wandb_project = 'sinhala-gpt'
wandb_run_name = 'sinhala-17m'

# data
dataset = 'sinhala'
gradient_accumulation_steps = 8
batch_size = 64
block_size = 512

# model (~17M params)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

# optimizer
learning_rate = 6e-4
max_iters = 100000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 100000
min_lr = 6e-5

# system
device = 'cuda'
dtype = 'bfloat16'
compile = True
