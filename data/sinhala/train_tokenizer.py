"""
Train a SentencePiece BPE tokenizer on the Sinhala MADLAD/CulturaX dataset.

This script:
1. Loads the HuggingFace dataset (polyglots/MADLAD_CulturaX_cleaned)
2. Extracts all Sinhala text into a temporary corpus file
3. Trains a SentencePiece BPE tokenizer with vocab_size=16000
4. Saves the tokenizer model and vocab to data/sinhala/

Usage:
    python data/sinhala/train_tokenizer.py
"""

import os
import sys
import time
import logging
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'train_tokenizer.log'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

def main():
    t_start = time.time()
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    VOCAB_SIZE = 16000
    MODEL_TYPE = 'bpe'
    MODEL_PREFIX = os.path.join(os.path.dirname(__file__), 'sinhala_tokenizer')
    CORPUS_FILE = os.path.join(os.path.dirname(__file__), 'corpus.txt')
    CHARACTER_COVERAGE = 0.9999  # High coverage for Sinhala Unicode
    NUM_THREADS = os.cpu_count() or 4
    
    logger.info("=" * 70)
    logger.info("SINHALA TOKENIZER TRAINING")
    logger.info("=" * 70)
    logger.info(f"Vocab size: {VOCAB_SIZE}")
    logger.info(f"Model type: {MODEL_TYPE}")
    logger.info(f"Output prefix: {MODEL_PREFIX}")
    logger.info(f"Character coverage: {CHARACTER_COVERAGE}")
    logger.info(f"Threads: {NUM_THREADS}")
    
    # -------------------------------------------------------------------------
    # Step 1: Load dataset from HuggingFace
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 1: Loading dataset from HuggingFace...")
    logger.info("-" * 70)
    
    from datasets import load_dataset
    
    t0 = time.time()
    ds = load_dataset("polyglots/MADLAD_CulturaX_cleaned")
    t1 = time.time()
    
    num_rows = len(ds['train'])
    logger.info(f"Dataset loaded in {t1 - t0:.1f}s")
    logger.info(f"Total rows: {num_rows:,}")
    logger.info(f"Features: {ds['train'].column_names}")
    
    # -------------------------------------------------------------------------
    # Step 2: Write text corpus to file
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 2: Writing text corpus to disk...")
    logger.info("-" * 70)
    
    t0 = time.time()
    total_chars = 0
    lines_written = 0
    
    with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
        for i, example in enumerate(ds['train']):
            text = example['text'].strip()
            if text:
                f.write(text + '\n')
                total_chars += len(text)
                lines_written += 1
            
            if (i + 1) % 1_000_000 == 0:
                logger.info(f"  Processed {i + 1:,} / {num_rows:,} rows ({100*(i+1)/num_rows:.1f}%)")
    
    t1 = time.time()
    corpus_size_mb = os.path.getsize(CORPUS_FILE) / (1024 * 1024)
    logger.info(f"Corpus written in {t1 - t0:.1f}s")
    logger.info(f"Lines written: {lines_written:,}")
    logger.info(f"Total characters: {total_chars:,}")
    logger.info(f"Corpus file size: {corpus_size_mb:.1f} MB")
    
    # -------------------------------------------------------------------------
    # Step 3: Train SentencePiece tokenizer
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 3: Training SentencePiece BPE tokenizer...")
    logger.info("-" * 70)
    
    import sentencepiece as spm
    
    t0 = time.time()
    
    spm.SentencePieceTrainer.train(
        input=CORPUS_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,
        character_coverage=CHARACTER_COVERAGE,
        num_threads=NUM_THREADS,
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Training parameters for quality
        byte_fallback=True,           # Handle unknown characters gracefully
        split_digits=True,            # Better number handling
        max_sentence_length=16384,    # Allow longer sentences
        shuffle_input_sentence=True,  # Better training
        input_sentence_size=10_000_000,  # Use up to 10M sentences for training
    )
    
    t1 = time.time()
    logger.info(f"Tokenizer trained in {t1 - t0:.1f}s")
    
    # -------------------------------------------------------------------------
    # Step 4: Verify tokenizer
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 4: Verifying tokenizer...")
    logger.info("-" * 70)
    
    sp = spm.SentencePieceProcessor()
    sp.load(MODEL_PREFIX + '.model')
    
    actual_vocab_size = sp.get_piece_size()
    logger.info(f"Actual vocab size: {actual_vocab_size}")
    
    # Test encode/decode roundtrip with Sinhala text
    test_texts = [
        "ආණ්ඩුව පටන් ගත් වහා ම ආණ්ඩුවෙන් බේරීමට පොලිස් පරීක්ෂක නිශාන්ත සිල්වා රට හැර ගියේ ය.",
        "ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දූපත් රාජ්‍යයකි.",
        "මම සිංහල කතා කරමි.",
    ]
    
    for text in test_texts:
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)
        pieces = sp.encode(text, out_type=str)
        logger.info(f"  Original:  {text}")
        logger.info(f"  Token IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")
        logger.info(f"  Pieces:    {pieces[:15]}{'...' if len(pieces) > 15 else ''}")
        logger.info(f"  Decoded:   {decoded}")
        logger.info(f"  Roundtrip OK: {text == decoded}")
        logger.info("")
    
    # -------------------------------------------------------------------------
    # Step 5: Cleanup and summary
    # -------------------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 5: Cleanup...")
    logger.info("-" * 70)
    
    # Remove the large corpus file to save disk space
    if os.path.exists(CORPUS_FILE):
        os.remove(CORPUS_FILE)
        logger.info(f"Removed temporary corpus file: {CORPUS_FILE}")
    
    model_path = MODEL_PREFIX + '.model'
    vocab_path = MODEL_PREFIX + '.vocab'
    logger.info(f"Tokenizer model saved: {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")
    logger.info(f"Tokenizer vocab saved: {vocab_path} ({os.path.getsize(vocab_path) / 1024:.1f} KB)")
    
    t_end = time.time()
    logger.info("=" * 70)
    logger.info(f"TOKENIZER TRAINING COMPLETE in {t_end - t_start:.1f}s")
    logger.info(f"Vocab size: {actual_vocab_size}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
