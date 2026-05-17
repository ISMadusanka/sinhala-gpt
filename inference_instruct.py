"""
Instruction-aware inference for the fine-tuned Sinhala GPT model.

Formats the user's question using the instruction template, generates a
response, and extracts only the assistant's reply.

Usage:
    python inference_instruct.py
    python inference_instruct.py --prompt="ශ්‍රී ලංකාව ගැන විස්තර කරන්න."
    python inference_instruct.py --prompt="දත්ත සැකසීම යනු කුමක්ද?" --max_tokens=300
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

# Must match the template used during fine-tuning data preparation
INSTRUCTION_PREFIX = "<|user|> {input} <|assistant|> "
RESPONSE_END_MARKER = "<|end|>"


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


def generate_instruction_response(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    """Generate a response to an instruction prompt."""
    # Format the prompt using the instruction template
    formatted_prompt = INSTRUCTION_PREFIX.format(input=prompt.strip())

    # Encode the formatted prompt
    input_ids = tokenizer.encode(formatted_prompt, out_type=int)

    logger.info(f"User prompt: '{prompt}'")
    logger.info(f"Formatted prompt: '{formatted_prompt}'")
    logger.info(f"Prompt tokens: {len(input_ids)}")
    logger.info(f"Generating up to {max_new_tokens} tokens (temp={temperature}, top_k={top_k})...")

    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generate
    t0 = time.time()
    with torch.no_grad():
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        if device_type == 'cuda':
            ctx = torch.amp.autocast(device_type=device_type, dtype=dtype)
        else:
            ctx = torch.amp.autocast(device_type='cpu', dtype=torch.float32)
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    t1 = time.time()

    # Decode the full output
    output_ids = y[0].tolist()
    full_text = tokenizer.decode(output_ids)

    # Extract only the generated part (after the prompt)
    generated_ids = output_ids[len(input_ids):]

    # Stop at EOS token if present
    eos_id = tokenizer.eos_id()
    if eos_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(eos_id)]

    generated_text = tokenizer.decode(generated_ids)

    # Remove the end marker if present in text
    if RESPONSE_END_MARKER in generated_text:
        generated_text = generated_text[:generated_text.index(RESPONSE_END_MARKER)]

    generated_text = generated_text.strip()

    tokens_per_sec = len(generated_ids) / (t1 - t0) if (t1 - t0) > 0 else 0
    logger.info(f"Generated {len(generated_ids)} tokens in {t1 - t0:.2f}s ({tokens_per_sec:.1f} tok/s)")

    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Sinhala GPT Instruction Inference')
    parser.add_argument('--checkpoint_dir', type=str, default='out-sinhala-instruct',
                        help='Directory containing fine-tuned ckpt.pt')
    parser.add_argument('--tokenizer', type=str, default='data/sinhala/sinhala_tokenizer.model',
                        help='Path to SentencePiece tokenizer model')
    parser.add_argument('--prompt', type=str, default='ශ්‍රී ලංකාව ගැන විස්තර කරන්න.',
                        help='Instruction/question in Sinhala')
    parser.add_argument('--max_tokens', type=int, default=300,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (lower = more focused)')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of responses to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode (type questions)')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SINHALA GPT — INSTRUCTION INFERENCE")
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

    # Load tokenizer and model
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.checkpoint_dir, device)

    # Interactive mode
    if args.interactive:
        logger.info("-" * 70)
        logger.info("INTERACTIVE MODE — Type your question in Sinhala (or 'exit' to quit)")
        logger.info("-" * 70)

        while True:
            try:
                prompt = input("\nYou: ").strip()
                if prompt.lower() in ('exit', 'quit', 'q'):
                    logger.info("Exiting interactive mode.")
                    break
                if not prompt:
                    continue

                response = generate_instruction_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=device,
                )
                print(f"\nAssistant: {response}")

            except KeyboardInterrupt:
                logger.info("\nExiting interactive mode.")
                break
    else:
        # Single/batch mode
        logger.info("-" * 70)
        logger.info(f"Generating {args.num_samples} response(s)...")
        logger.info("-" * 70)

        for i in range(args.num_samples):
            if args.num_samples > 1:
                logger.info(f"\n{'='*40} Response {i+1}/{args.num_samples} {'='*40}")

            response = generate_instruction_response(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
            )

            print(f"\n  Question: {args.prompt}")
            print(f"  Answer:   {response}")
            print()

    logger.info("Inference complete!")


if __name__ == '__main__':
    main()
