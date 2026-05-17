"""
Inference script for the trained Sinhala GPT model.

Loads the trained checkpoint and SentencePiece tokenizer, then generates
Sinhala text from a given prompt.

Usage:
    python inference.py
    python inference.py --prompt="ශ්‍රී ලංකාව"
    python inference.py --prompt="මම" --max_tokens=200 --temperature=0.8
    python inference.py --num_samples=5
"""

import os
import sys
import time
import argparse
import logging

import torch
import sentencepiece as spm

from model import GPTConfig, GPT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_dir, device):
    """Load model from checkpoint."""
    ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
    logger.info(f"Loading checkpoint from: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']

    logger.info(f"Model config: {model_args}")

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    # Remove compiled model prefix if present
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    iter_num = checkpoint.get('iter_num', 'unknown')
    best_val_loss = checkpoint.get('best_val_loss', 'unknown')
    logger.info(f"Checkpoint loaded: iter={iter_num}, best_val_loss={best_val_loss}")

    return model


def load_tokenizer(tokenizer_path):
    """Load SentencePiece tokenizer."""
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    logger.info(f"Tokenizer loaded. Vocab size: {sp.get_piece_size()}")
    return sp


def generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    """Generate text from a prompt."""
    # Encode prompt
    if prompt:
        input_ids = tokenizer.encode(prompt, out_type=int)
    else:
        # Use BOS token if no prompt
        input_ids = [tokenizer.bos_id()]

    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"Prompt token IDs: {input_ids}")
    logger.info(f"Generating {max_new_tokens} tokens (temp={temperature}, top_k={top_k})...")

    # Generate
    t0 = time.time()
    with torch.no_grad():
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        ctx = torch.amp.autocast(device_type=device_type, dtype=dtype) if device_type == 'cuda' else torch.autocast('cpu')
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    t1 = time.time()

    # Decode output
    output_ids = y[0].tolist()
    generated_ids = output_ids[len(input_ids):]  # Only the new tokens

    # Stop at EOS token if present
    eos_id = tokenizer.eos_id()
    if eos_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(eos_id)]

    generated_text = tokenizer.decode(generated_ids)
    full_text = tokenizer.decode(output_ids)

    tokens_per_sec = len(generated_ids) / (t1 - t0) if (t1 - t0) > 0 else 0
    logger.info(f"Generated {len(generated_ids)} tokens in {t1 - t0:.2f}s ({tokens_per_sec:.1f} tok/s)")

    return full_text, generated_text


def main():
    parser = argparse.ArgumentParser(description='Sinhala GPT Inference')
    parser.add_argument('--checkpoint_dir', type=str, default='out-sinhala',
                        help='Directory containing ckpt.pt')
    parser.add_argument('--tokenizer', type=str, default='data/sinhala/sinhala_tokenizer.model',
                        help='Path to SentencePiece tokenizer model')
    parser.add_argument('--prompt', type=str, default='ශ්‍රී ලංකාව',
                        help='Input prompt in Sinhala')
    parser.add_argument('--max_tokens', type=int, default=200,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SINHALA GPT INFERENCE")
    logger.info("=" * 70)

    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    logger.info(f"Device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Set seed
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    # Load model
    model = load_model(args.checkpoint_dir, device)

    # Generate samples
    logger.info("-" * 70)
    logger.info(f"Generating {args.num_samples} samples...")
    logger.info("-" * 70)

    for i in range(args.num_samples):
        logger.info(f"\n{'='*40} Sample {i+1}/{args.num_samples} {'='*40}")
        full_text, generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(f"\n--- Sample {i+1} ---")
        print(full_text)
        print(f"{'='*70}\n")

    logger.info("Inference complete!")


if __name__ == '__main__':
    main()
