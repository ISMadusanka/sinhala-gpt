"""
Prepare the Aya instruction dataset for fine-tuning Sinhala GPT.

This script:
1. Loads the CohereLabs/aya_dataset from HuggingFace
2. Filters for Sinhala (language_code == "sin")
3. Formats each example as: <|user|> {input} <|assistant|> {target} <|end|>
4. Tokenizes with the pre-trained SentencePiece tokenizer
5. Writes train.bin, val.bin, meta.pkl
6. Prepares the pre-trained checkpoint for fine-tuning

Prerequisites:
    - Trained tokenizer at data/sinhala/sinhala_tokenizer.model
    - Pre-trained model checkpoint at out-sinhala/ckpt.pt

Usage:
    python data/sinhala_instruct/prepare.py
"""

import os
import sys
import time
import pickle
import logging
import shutil

import numpy as np
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'prepare.log'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

# Instruction format template
INSTRUCTION_TEMPLATE = "<|user|> {input} <|assistant|> {target} <|end|>"


def format_instruction(inputs, targets):
    """Format an instruction-response pair into the training format."""
    return INSTRUCTION_TEMPLATE.format(input=inputs.strip(), target=targets.strip())


def main():
    t_start = time.time()

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    DATA_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(DATA_DIR))
    TOKENIZER_MODEL = os.path.join(PROJECT_ROOT, 'data', 'sinhala', 'sinhala_tokenizer.model')
    PRETRAINED_CKPT = os.path.join(PROJECT_ROOT, 'out-sinhala', 'ckpt.pt')
    FINETUNE_OUT_DIR = os.path.join(PROJECT_ROOT, 'out-sinhala-instruct')
    VAL_FRACTION = 0.05  # 5% for validation (small dataset, need more val data)

    logger.info("=" * 70)
    logger.info("SINHALA INSTRUCTION FINE-TUNING DATA PREPARATION")
    logger.info("=" * 70)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Tokenizer: {TOKENIZER_MODEL}")
    logger.info(f"Pre-trained checkpoint: {PRETRAINED_CKPT}")
    logger.info(f"Fine-tune output dir: {FINETUNE_OUT_DIR}")

    # -------------------------------------------------------------------------
    # Step 1: Load tokenizer
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 1: Loading SentencePiece tokenizer...")
    logger.info("-" * 70)

    import sentencepiece as spm

    if not os.path.exists(TOKENIZER_MODEL):
        logger.error(f"Tokenizer not found: {TOKENIZER_MODEL}")
        logger.error("Run data/sinhala/train_tokenizer.py first!")
        sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    vocab_size = sp.get_piece_size()
    eos_id = sp.eos_id()

    logger.info(f"Tokenizer loaded. Vocab size: {vocab_size}, EOS ID: {eos_id}")

    # -------------------------------------------------------------------------
    # Step 2: Load and filter dataset
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 2: Loading Aya dataset and filtering Sinhala...")
    logger.info("-" * 70)

    from datasets import load_dataset

    t0 = time.time()
    ds = load_dataset("CohereLabs/aya_dataset")
    t1 = time.time()
    logger.info(f"Aya dataset loaded in {t1 - t0:.1f}s")

    # Filter for Sinhala
    t0 = time.time()
    sinhala_ds = ds.filter(lambda example: example["language_code"] == "sin")
    t1 = time.time()

    num_train = len(sinhala_ds['train'])
    num_test = len(sinhala_ds['test']) if 'test' in sinhala_ds else 0
    logger.info(f"Sinhala filtering done in {t1 - t0:.1f}s")
    logger.info(f"Sinhala train rows: {num_train:,}")
    logger.info(f"Sinhala test rows: {num_test:,}")

    # -------------------------------------------------------------------------
    # Step 3: Format instruction pairs
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 3: Formatting instruction pairs...")
    logger.info("-" * 70)

    # Show a few examples
    for i in range(min(3, num_train)):
        sample = sinhala_ds['train'][i]
        formatted = format_instruction(sample['inputs'], sample['targets'])
        logger.info(f"  Example {i+1}:")
        logger.info(f"    Input:  {sample['inputs'][:80]}...")
        logger.info(f"    Target: {sample['targets'][:80]}...")
        logger.info(f"    Formatted (first 120 chars): {formatted[:120]}...")
        logger.info("")

    # -------------------------------------------------------------------------
    # Step 4: Split and tokenize
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 4: Splitting and tokenizing...")
    logger.info("-" * 70)

    # Use train split; create our own val split from it
    train_data = sinhala_ds['train']

    # Split into train/val
    split = train_data.train_test_split(test_size=VAL_FRACTION, seed=42, shuffle=True)
    train_split = split['train']
    val_split = split['test']

    logger.info(f"Train examples: {len(train_split):,}")
    logger.info(f"Val examples: {len(val_split):,}")

    # Tokenize
    def tokenize_instruction(example):
        formatted = format_instruction(example['inputs'], example['targets'])
        ids = sp.encode(formatted, out_type=int)
        ids.append(eos_id)
        return {'ids': ids, 'len': len(ids)}

    t0 = time.time()
    splits = {'train': train_split, 'val': val_split}
    tokenized = {}

    for split_name, split_data in splits.items():
        tokenized[split_name] = split_data.map(
            tokenize_instruction,
            remove_columns=split_data.column_names,
            desc=f"Tokenizing {split_name}",
        )
    t1 = time.time()

    logger.info(f"Tokenization completed in {t1 - t0:.1f}s")

    for split_name in ['train', 'val']:
        total_tokens = sum(tokenized[split_name]['len'])
        avg_len = total_tokens / len(tokenized[split_name]) if len(tokenized[split_name]) > 0 else 0
        logger.info(f"  {split_name}: {total_tokens:,} tokens, avg {avg_len:.1f} tokens/example")

    # -------------------------------------------------------------------------
    # Step 5: Write binary files
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 5: Writing binary files...")
    logger.info("-" * 70)

    for split_name, dset in tokenized.items():
        t0 = time.time()

        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(DATA_DIR, f'{split_name}.bin')
        dtype = np.uint16

        logger.info(f"Writing {split_name}.bin: {arr_len:,} tokens")

        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        idx = 0
        for example in tqdm(dset, desc=f'Writing {split_name}.bin'):
            ids = np.array(example['ids'], dtype=np.uint16)
            arr[idx : idx + len(ids)] = ids
            idx += len(ids)

        arr.flush()
        t1 = time.time()

        file_size_kb = os.path.getsize(filename) / 1024
        logger.info(f"  {split_name}.bin: {file_size_kb:.1f} KB, written in {t1 - t0:.1f}s")

    # -------------------------------------------------------------------------
    # Step 6: Save meta.pkl
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 6: Saving meta.pkl...")
    logger.info("-" * 70)

    meta = {
        'vocab_size': vocab_size,
        'tokenizer_model': TOKENIZER_MODEL,
        'eos_id': eos_id,
        'bos_id': sp.bos_id(),
        'pad_id': sp.pad_id(),
        'unk_id': sp.unk_id(),
        'instruction_template': INSTRUCTION_TEMPLATE,
    }

    meta_path = os.path.join(DATA_DIR, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    logger.info(f"meta.pkl saved: vocab_size={vocab_size}")

    # -------------------------------------------------------------------------
    # Step 7: Prepare pre-trained checkpoint for fine-tuning
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 7: Preparing checkpoint for fine-tuning...")
    logger.info("-" * 70)

    if not os.path.exists(PRETRAINED_CKPT):
        logger.warning(f"Pre-trained checkpoint not found: {PRETRAINED_CKPT}")
        logger.warning("You can still fine-tune, but you'll need to train from scratch.")
        logger.warning("Run: python train.py config/finetune_sinhala.py --init_from=scratch")
    else:
        os.makedirs(FINETUNE_OUT_DIR, exist_ok=True)
        finetune_ckpt_path = os.path.join(FINETUNE_OUT_DIR, 'ckpt.pt')

        logger.info(f"Loading pre-trained checkpoint: {PRETRAINED_CKPT}")
        checkpoint = torch.load(PRETRAINED_CKPT, map_location='cpu')

        original_iter = checkpoint.get('iter_num', 'unknown')
        original_loss = checkpoint.get('best_val_loss', 'unknown')
        logger.info(f"  Pre-trained iter: {original_iter}")
        logger.info(f"  Pre-trained best val loss: {original_loss}")

        # Reset training state for fine-tuning (keep model weights, reset optimizer)
        checkpoint['iter_num'] = 0
        checkpoint['best_val_loss'] = 1e9

        # Update config to point to instruction dataset
        if 'config' in checkpoint:
            checkpoint['config']['dataset'] = 'sinhala_instruct'

        # Save the fine-tuning checkpoint
        torch.save(checkpoint, finetune_ckpt_path)

        ckpt_size_mb = os.path.getsize(finetune_ckpt_path) / (1024 * 1024)
        logger.info(f"  Fine-tuning checkpoint saved: {finetune_ckpt_path} ({ckpt_size_mb:.1f} MB)")
        logger.info(f"  iter_num reset to 0, best_val_loss reset to 1e9")
        logger.info(f"  Model weights preserved, optimizer state preserved (LR will be overridden)")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    t_end = time.time()
    logger.info("=" * 70)
    logger.info(f"INSTRUCTION DATA PREPARATION COMPLETE in {t_end - t_start:.1f}s")
    logger.info(f"  train.bin: {os.path.join(DATA_DIR, 'train.bin')}")
    logger.info(f"  val.bin:   {os.path.join(DATA_DIR, 'val.bin')}")
    logger.info(f"  meta.pkl:  {meta_path}")
    if os.path.exists(PRETRAINED_CKPT):
        logger.info(f"  Checkpoint: {os.path.join(FINETUNE_OUT_DIR, 'ckpt.pt')}")
    logger.info("")
    logger.info("Next step: python train.py config/finetune_sinhala.py")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
