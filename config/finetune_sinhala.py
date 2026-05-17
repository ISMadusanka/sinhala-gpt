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
gradient_accumulation_steps = 8  # effective batch = 8 * 16 * 512 = 65,536 tokens
batch_size = 16
block_size = 512

# model (must match pre-trained model — these are overridden by checkpoint anyway)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.35               # strong dropout to keep train/val gap minimal
bias = False

# init from pre-trained checkpoint (prepared by prepare.py)
init_from = 'resume'

# optimizer (lower learning rate for fine-tuning)
learning_rate = 5e-5         # 12x lower than pre-training to prevent overfitting
max_iters = 100000           # full training run
weight_decay = 2e-1          # stronger weight decay for regularization
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate schedule
decay_lr = True
warmup_iters = 1000          # longer warmup for 100K iters
lr_decay_iters = 100000      # decay over full training run
min_lr = 1e-5                # min LR

# system
device = 'cuda'
dtype = 'bfloat16'
compile = True
