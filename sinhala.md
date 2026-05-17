# Sinhala GPT — Training & Inference Guide

A ~17M parameter Sinhala language model built on the nanoGPT framework, pre-trained on the `polyglots/MADLAD_CulturaX_cleaned` dataset (10.7M rows of Sinhala text).

---

## Prerequisites

### Install Dependencies

```bash
pip install torch numpy datasets sentencepiece tqdm
```

### HuggingFace Login (required for dataset access)

```bash
huggingface-cli login
```

---

## Quick Start (Full Pipeline)

Run everything with a single command:

```bash
python run_pipeline.py
```

This will execute all 4 steps sequentially:
1. Train tokenizer
2. Prepare data
3. Train model
4. Run inference

---

## Step-by-Step Instructions

### Step 1: Train the Tokenizer

Trains a SentencePiece BPE tokenizer (16K vocab) on all Sinhala text.

```bash
python data/sinhala/train_tokenizer.py
```

**Output files:**
- `data/sinhala/sinhala_tokenizer.model`
- `data/sinhala/sinhala_tokenizer.vocab`

**Logs:** `data/sinhala/train_tokenizer.log`

---

### Step 2: Prepare the Data

Tokenizes all 10.7M rows and creates binary files for training.

```bash
python data/sinhala/prepare.py
```

**Output files:**
- `data/sinhala/train.bin` — Training data (99.5%)
- `data/sinhala/val.bin` — Validation data (0.5%)
- `data/sinhala/meta.pkl` — Metadata (vocab size, token IDs)

**Logs:** `data/sinhala/prepare.log`

---

### Step 3: Train the Model

Pre-trains the GPT model on the prepared data.

```bash
python train.py config/train_sinhala.py
```

**Model architecture (~17M parameters):**

| Config | Value |
|--------|-------|
| Layers | 6 |
| Heads | 6 |
| Embedding dim | 384 |
| Context length | 512 |
| Vocab size | 16,000 |
| Bias | False |
| Dropout | 0.0 |

**Training config:**

| Config | Value |
|--------|-------|
| Batch size | 64 |
| Gradient accumulation | 8 |
| Effective batch (tokens/step) | 262,144 |
| Learning rate | 6e-4 (cosine decay) |
| Warmup steps | 2,000 |
| Total iterations | 100,000 |
| Precision | bfloat16 |

**Output:** `out-sinhala/ckpt.pt`

---

### Step 4: Run Inference

Generate Sinhala text from the trained model.

```bash
python inference.py
```

**With custom options:**

```bash
# Custom prompt
python inference.py --prompt="ශ්‍රී ලංකාව"

# Adjust generation parameters
python inference.py --prompt="මම" --max_tokens=200 --temperature=0.8 --top_k=50

# Generate more samples
python inference.py --num_samples=5

# Use a different checkpoint
python inference.py --checkpoint_dir=out-sinhala --tokenizer=data/sinhala/sinhala_tokenizer.model
```

**Inference arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | `ශ්‍රී ලංකාව` | Input prompt in Sinhala |
| `--max_tokens` | 200 | Max tokens to generate |
| `--temperature` | 0.8 | Sampling temperature (lower = more deterministic) |
| `--top_k` | 50 | Top-k sampling |
| `--num_samples` | 3 | Number of samples to generate |
| `--checkpoint_dir` | `out-sinhala` | Checkpoint directory |
| `--tokenizer` | `data/sinhala/sinhala_tokenizer.model` | Tokenizer path |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

---

## Resume from a Specific Step

If a step has already completed, skip it:

```bash
# Resume from Step 3 (tokenizer + data already done)
python run_pipeline.py --start_step=3

# Run only specific steps
python run_pipeline.py --steps=1,2
python run_pipeline.py --steps=3
python run_pipeline.py --steps=4
```

---

## Resume Training from Checkpoint

If training was interrupted, resume from the last saved checkpoint:

```bash
python train.py config/train_sinhala.py --init_from=resume
```

---

## Project Structure

```
sinhala-gpt/
├── model.py                         # GPT model definition (unchanged)
├── train.py                         # Training loop (unchanged)
├── run_pipeline.py                  # Full pipeline runner
├── inference.py                     # Inference script
├── config/
│   └── train_sinhala.py             # Sinhala training config
├── data/
│   └── sinhala/
│       ├── train_tokenizer.py       # Tokenizer training script
│       ├── prepare.py               # Data preparation script
│       ├── sinhala_tokenizer.model  # (generated) Tokenizer model
│       ├── sinhala_tokenizer.vocab  # (generated) Tokenizer vocab
│       ├── train.bin                # (generated) Training data
│       ├── val.bin                  # (generated) Validation data
│       └── meta.pkl                 # (generated) Metadata
└── out-sinhala/
    └── ckpt.pt                      # (generated) Model checkpoint
```

---

## GPU Requirements

- **Minimum:** 1 GPU with 8GB VRAM
- **Recommended:** 1 GPU with 16GB+ VRAM (A100, H100, RTX 3090/4090)
- Training uses `bfloat16` mixed precision and `torch.compile` for maximum speed
- The model is only ~17M parameters, so it fits comfortably on any modern GPU
