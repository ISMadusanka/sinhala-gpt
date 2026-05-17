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

## Instruction Fine-tuning (Step 5-7)

After pre-training, you can fine-tune the model to follow instructions using the Aya dataset.

### Step 5: Prepare Instruction Data

Downloads the Aya dataset, filters Sinhala examples, formats them as instruction pairs, and prepares the pre-trained checkpoint for fine-tuning.

```bash
python data/sinhala_instruct/prepare.py
```

**Dataset:** `CohereLabs/aya_dataset` (~14,524 Sinhala instruction-response pairs)

**Instruction format used during training:**
```
<|user|> {question} <|assistant|> {answer} <|end|>
```

**Output files:**
- `data/sinhala_instruct/train.bin` — Training data (95%)
- `data/sinhala_instruct/val.bin` — Validation data (5%)
- `data/sinhala_instruct/meta.pkl` — Metadata
- `out-sinhala-instruct/ckpt.pt` — Pre-trained checkpoint prepared for fine-tuning

---

### Step 6: Fine-tune the Model

Fine-tunes the pre-trained model on instruction data with lower learning rate and dropout.

```bash
python train.py config/finetune_sinhala.py
python train.py config/finetune_sinhala.py --init_from=scratch

```

**Fine-tuning config differences vs pre-training:**

| Config | Pre-training | Fine-tuning |
|--------|-------------|-------------|
| Learning rate | 6e-4 | 1e-4 |
| Dropout | 0.0 | 0.1 |
| Max iterations | 100,000 | 5,000 |
| Batch size | 64 | 32 |
| Eval interval | 1,000 | 250 |
| Save checkpoint | Always | Only when val loss improves |

**Output:** `out-sinhala-instruct/ckpt.pt`

---

### Step 7: Run Instruction Inference

Generate responses to Sinhala questions using the fine-tuned model.

```bash
python inference_instruct.py --prompt="ශ්‍රී ලංකාව ගැන විස්තර කරන්න."
```

**Interactive chat mode:**
```bash
python inference_instruct.py --interactive
```

**All arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | `ශ්‍රී ලංකාව ගැන විස්තර කරන්න.` | Question in Sinhala |
| `--max_tokens` | 300 | Max tokens to generate |
| `--temperature` | 0.7 | Sampling temperature |
| `--top_k` | 40 | Top-k sampling |
| `--num_samples` | 1 | Number of responses |
| `--checkpoint_dir` | `out-sinhala-instruct` | Checkpoint directory |
| `--interactive` | (flag) | Run in interactive chat mode |

---

### Run All Fine-tuning Steps at Once

```bash
python run_pipeline.py --steps=5,6,7
```

---

## Project Structure

```
sinhala-gpt/
├── model.py                         # GPT model definition (unchanged)
├── train.py                         # Training loop (unchanged)
├── run_pipeline.py                  # Full pipeline runner (steps 1-7)
├── inference.py                     # Inference script (pre-trained)
├── inference_instruct.py            # Inference script (fine-tuned)
├── config/
│   ├── train_sinhala.py             # Pre-training config
│   └── finetune_sinhala.py          # Fine-tuning config
├── data/
│   ├── sinhala/
│   │   ├── train_tokenizer.py       # Tokenizer training script
│   │   ├── prepare.py               # Pre-training data preparation
│   │   ├── sinhala_tokenizer.model  # (generated) Tokenizer model
│   │   ├── sinhala_tokenizer.vocab  # (generated) Tokenizer vocab
│   │   ├── train.bin                # (generated) Pre-training data
│   │   ├── val.bin                  # (generated) Validation data
│   │   └── meta.pkl                 # (generated) Metadata
│   └── sinhala_instruct/
│       ├── prepare.py               # Instruction data preparation
│       ├── train.bin                # (generated) Instruction train data
│       ├── val.bin                  # (generated) Instruction val data
│       └── meta.pkl                 # (generated) Metadata
├── out-sinhala/
│   └── ckpt.pt                      # (generated) Pre-trained checkpoint
└── out-sinhala-instruct/
    └── ckpt.pt                      # (generated) Fine-tuned checkpoint
```

---

## GPU Requirements

- **Minimum:** 1 GPU with 8GB VRAM
- **Recommended:** 1 GPU with 16GB+ VRAM (A100, H100, RTX 3090/4090)
- Training uses `bfloat16` mixed precision and `torch.compile` for maximum speed
- The model is only ~17M parameters, so it fits comfortably on any modern GPU


## NOTES
~10.0: Completely random guessing (Untrained).
~7.0 - 8.0: The model is starting to learn basic character/token frequencies (e.g., space is common, rare characters are rare).
~4.0 - 5.0: The model has learned basic Sinhala word structures and common short words, but sentences will still look like gibberish.
~3.0 - 3.5: Sentences start to look like real Sinhala. The grammar might be slightly broken, but it's generating recognizable phrases.
~2.5 - 2.8: This is typically considered a good, usable model for this size. It will generate coherent Sinhala sentences, have basic grammar, and understand context reasonably well. (For reference, the original English GPT-2 124M model achieved a loss of about 3.11, and down to 2.85 on similar datasets).
< 2.0: The model is exceptionally good at predicting the exact text (though if it gets too low, like < 1.0, it might mean the model is just memorizing the training data, known as overfitting).