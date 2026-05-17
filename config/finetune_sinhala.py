# Fine-tuning config for Sinhala GPT on instruction data (Aya dataset)
#
# This loads the pre-trained Sinhala model and fine-tunes it on
# instruction-following data (input/target pairs).
#
# Prerequisites:
#   1. Pre-trained model at out-sinhala/ckpt.pt
#   2. Run: python data/sinhala_instruct/prepare.py
#
# Usage:
#   python train.py config/finetune_sinhala.py

# I/O
out_dir = 'out-sinhala-instruct'
eval_interval = 250          # evaluate more frequently (small dataset)
log_interval = 10
eval_iters = 100             # fewer eval iters (small val set)
eval_only = False
always_save_checkpoint = False  # only save when val loss improves

# wandb logging
wandb_log = False
wandb_project = 'sinhala-gpt'
wandb_run_name = 'sinhala-instruct-finetune'

# data
dataset = 'sinhala_instruct'
gradient_accumulation_steps = 4  # smaller effective batch for fine-tuning
batch_size = 32
block_size = 512

# model (must match pre-trained model — these are overridden by checkpoint anyway)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1                # add some dropout for fine-tuning (prevents overfitting)
bias = False

# init from pre-trained checkpoint (prepared by prepare.py)
init_from = 'resume'

# optimizer (lower learning rate for fine-tuning)
learning_rate = 1e-4         # 6x lower than pre-training
max_iters = 5000             # much fewer iterations (small dataset)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate schedule
decay_lr = True
warmup_iters = 200           # shorter warmup
lr_decay_iters = 5000        # decay over full fine-tuning run
min_lr = 1e-5                # min LR

# system
device = 'cuda'
dtype = 'bfloat16'
compile = True
