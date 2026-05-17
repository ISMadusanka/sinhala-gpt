"""
Prepare the Sinhala MADLAD/CulturaX dataset for nanoGPT training.

This script:
1. Loads the HuggingFace dataset
2. Tokenizes all text using the trained SentencePiece tokenizer
3. Splits into train (99.5%) and val (0.5%)
4. Writes train.bin and val.bin as uint16 memmap files
5. Saves meta.pkl with vocab_size for nanoGPT auto-detection

Prerequisites:
    Run train_tokenizer.py first to create the tokenizer model.

Usage:
    python data/sinhala/prepare.py
"""

import os
import sys
import time
import pickle
import logging

import numpy as np
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


def main():
    t_start = time.time()
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    DATA_DIR = os.path.dirname(__file__)
    TOKENIZER_MODEL = os.path.join(DATA_DIR, 'sinhala_tokenizer.model')
    VAL_FRACTION = 0.005  # 0.5% for validation
    NUM_PROC = os.cpu_count() or 4
    
    logger.info("=" * 70)
    logger.info("SINHALA DATA PREPARATION FOR NANOGPT")
    logger.info("=" * 70)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Tokenizer model: {TOKENIZER_MODEL}")
    logger.info(f"Validation fraction: {VAL_FRACTION}")
    logger.info(f"Num processors: {NUM_PROC}")
    
    # -------------------------------------------------------------------------
    # Step 1: Load tokenizer
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 1: Loading SentencePiece tokenizer...")
    logger.info("-" * 70)
    
    import sentencepiece as spm
    
    if not os.path.exists(TOKENIZER_MODEL):
        logger.error(f"Tokenizer model not found: {TOKENIZER_MODEL}")
        logger.error("Run train_tokenizer.py first!")
        sys.exit(1)
    
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    vocab_size = sp.get_piece_size()
    eos_id = sp.eos_id()
    
    logger.info(f"Tokenizer loaded. Vocab size: {vocab_size}")
    logger.info(f"EOS token ID: {eos_id}")
    
    assert vocab_size < 2**16, f"Vocab size {vocab_size} too large for uint16 storage!"
    
    # -------------------------------------------------------------------------
    # Step 2: Load dataset
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 2: Loading dataset from HuggingFace...")
    logger.info("-" * 70)
    
    from datasets import load_dataset
    
    t0 = time.time()
    ds = load_dataset("polyglots/MADLAD_CulturaX_cleaned")
    t1 = time.time()
    
    num_rows = len(ds['train'])
    logger.info(f"Dataset loaded in {t1 - t0:.1f}s")
    logger.info(f"Total rows: {num_rows:,}")
    
    # -------------------------------------------------------------------------
    # Step 3: Split into train and val
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 3: Splitting dataset into train/val...")
    logger.info("-" * 70)
    
    t0 = time.time()
    split_dataset = ds['train'].train_test_split(
        test_size=VAL_FRACTION,
        seed=2357,
        shuffle=True,
    )
    split_dataset['val'] = split_dataset.pop('test')
    t1 = time.time()
    
    logger.info(f"Split completed in {t1 - t0:.1f}s")
    logger.info(f"Train rows: {len(split_dataset['train']):,}")
    logger.info(f"Val rows:   {len(split_dataset['val']):,}")
    
    # -------------------------------------------------------------------------
    # Step 4: Tokenize the dataset
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 4: Tokenizing dataset...")
    logger.info("-" * 70)
    
    def tokenize_function(example):
        """Tokenize a single example and append EOS token."""
        text = example['text'].strip()
        if text:
            ids = sp.encode(text, out_type=int)
            ids.append(eos_id)  # Append end-of-sequence token
        else:
            ids = []
        return {'ids': ids, 'len': len(ids)}
    
    t0 = time.time()
    tokenized = split_dataset.map(
        tokenize_function,
        remove_columns=split_dataset['train'].column_names,
        desc="Tokenizing",
        num_proc=NUM_PROC,
    )
    t1 = time.time()
    
    logger.info(f"Tokenization completed in {t1 - t0:.1f}s")
    
    # Log some stats
    for split_name in ['train', 'val']:
        total_tokens = sum(tokenized[split_name]['len'])
        avg_len = total_tokens / len(tokenized[split_name]) if len(tokenized[split_name]) > 0 else 0
        logger.info(f"  {split_name}: {total_tokens:,} tokens, avg {avg_len:.1f} tokens/doc")
    
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
        dtype = np.uint16  # Safe since vocab_size < 2^16
        
        logger.info(f"Writing {split_name}.bin: {arr_len:,} tokens ({arr_len * 2 / (1024**3):.2f} GB)")
        
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = min(1024, len(dset))  # Don't exceed dataset size
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {split_name}.bin'):
            batch = dset.shard(
                num_shards=total_batches,
                index=batch_idx,
                contiguous=True,
            ).with_format('numpy')
            
            # Concatenate all token IDs in this batch
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        
        arr.flush()
        t1 = time.time()
        
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        logger.info(f"  {split_name}.bin written in {t1 - t0:.1f}s ({file_size_mb:.1f} MB)")
    
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
    }
    
    meta_path = os.path.join(DATA_DIR, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    
    logger.info(f"meta.pkl saved: {meta}")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    t_end = time.time()
    logger.info("=" * 70)
    logger.info(f"DATA PREPARATION COMPLETE in {t_end - t_start:.1f}s")
    logger.info(f"  train.bin: {os.path.join(DATA_DIR, 'train.bin')}")
    logger.info(f"  val.bin:   {os.path.join(DATA_DIR, 'val.bin')}")
    logger.info(f"  meta.pkl:  {meta_path}")
    logger.info(f"  Vocab size: {vocab_size}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
