"""
Sinhala GPT — Full Training Pipeline Runner

Orchestrates the entire pipeline:
  Step 1: Train SentencePiece tokenizer
  Step 2: Prepare data (tokenize + create binary files)
  Step 3: Train the GPT model
  Step 4: Run inference

Usage:
    python run_pipeline.py                  # Run all steps
    python run_pipeline.py --start_step=3   # Resume from step 3
    python run_pipeline.py --steps=1,2      # Run only specific steps
"""

import os
import sys
import time
import argparse
import subprocess
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log', mode='a'),
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_step(step_num, name, cmd, cwd=None):
    """Run a pipeline step as a subprocess."""
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"STEP {step_num}: {name}")
    logger.info("=" * 70)
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Working directory: {cwd or PROJECT_ROOT}")
    logger.info("-" * 70)

    t0 = time.time()

    process = subprocess.Popen(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
    )
    process.wait()

    t1 = time.time()
    elapsed = t1 - t0

    if process.returncode != 0:
        logger.error(f"STEP {step_num} FAILED with return code {process.returncode}")
        logger.error(f"Elapsed time: {elapsed:.1f}s")
        sys.exit(process.returncode)

    logger.info("-" * 70)
    logger.info(f"STEP {step_num} COMPLETED in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 70)

    return elapsed


def main():
    parser = argparse.ArgumentParser(description='Sinhala GPT Training Pipeline')
    parser.add_argument('--start_step', type=int, default=1,
                        help='Start from this step (1-4)')
    parser.add_argument('--steps', type=str, default=None,
                        help='Comma-separated list of steps to run (e.g., "1,2,3")')
    parser.add_argument('--config', type=str, default='config/train_sinhala.py',
                        help='Training config file')
    parser.add_argument('--inference_prompt', type=str, default='ශ්‍රී ලංකාව',
                        help='Prompt for inference step')
    args = parser.parse_args()

    # Determine which steps to run
    if args.steps:
        steps_to_run = set(int(s.strip()) for s in args.steps.split(','))
    else:
        steps_to_run = set(range(args.start_step, 5))  # steps 1-4

    logger.info("=" * 70)
    logger.info("SINHALA GPT — FULL TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Steps to run: {sorted(steps_to_run)}")
    logger.info(f"Training config: {args.config}")

    total_start = time.time()
    step_times = {}

    python = sys.executable  # Use the same Python interpreter

    # -------------------------------------------------------------------------
    # Step 1: Train Tokenizer
    # -------------------------------------------------------------------------
    if 1 in steps_to_run:
        elapsed = run_step(
            step_num=1,
            name="TRAIN TOKENIZER",
            cmd=[python, os.path.join('data', 'sinhala', 'train_tokenizer.py')],
        )
        step_times[1] = elapsed

        # Verify outputs
        tokenizer_model = os.path.join(PROJECT_ROOT, 'data', 'sinhala', 'sinhala_tokenizer.model')
        if not os.path.exists(tokenizer_model):
            logger.error(f"Tokenizer model not found at {tokenizer_model}")
            sys.exit(1)
        logger.info(f"✓ Tokenizer model verified: {tokenizer_model}")

    # -------------------------------------------------------------------------
    # Step 2: Prepare Data
    # -------------------------------------------------------------------------
    if 2 in steps_to_run:
        elapsed = run_step(
            step_num=2,
            name="PREPARE DATA",
            cmd=[python, os.path.join('data', 'sinhala', 'prepare.py')],
        )
        step_times[2] = elapsed

        # Verify outputs
        for fname in ['train.bin', 'val.bin', 'meta.pkl']:
            fpath = os.path.join(PROJECT_ROOT, 'data', 'sinhala', fname)
            if not os.path.exists(fpath):
                logger.error(f"Missing expected output: {fpath}")
                sys.exit(1)
            size = os.path.getsize(fpath)
            logger.info(f"✓ {fname} verified ({size / (1024*1024):.1f} MB)")

    # -------------------------------------------------------------------------
    # Step 3: Train Model
    # -------------------------------------------------------------------------
    if 3 in steps_to_run:
        elapsed = run_step(
            step_num=3,
            name="TRAIN MODEL",
            cmd=[python, 'train.py', args.config],
        )
        step_times[3] = elapsed

        # Verify checkpoint
        ckpt_path = os.path.join(PROJECT_ROOT, 'out-sinhala', 'ckpt.pt')
        if os.path.exists(ckpt_path):
            size = os.path.getsize(ckpt_path)
            logger.info(f"✓ Checkpoint verified: {ckpt_path} ({size / (1024*1024):.1f} MB)")
        else:
            logger.warning(f"Checkpoint not found at {ckpt_path} (training may still be in progress)")

    # -------------------------------------------------------------------------
    # Step 4: Run Inference
    # -------------------------------------------------------------------------
    if 4 in steps_to_run:
        elapsed = run_step(
            step_num=4,
            name="RUN INFERENCE",
            cmd=[
                python, 'inference.py',
                f'--prompt={args.inference_prompt}',
                '--num_samples=3',
                '--max_tokens=200',
            ],
        )
        step_times[4] = elapsed

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total_elapsed = time.time() - total_start

    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)

    for step_num in sorted(step_times.keys()):
        step_names = {1: 'Train Tokenizer', 2: 'Prepare Data', 3: 'Train Model', 4: 'Inference'}
        logger.info(f"  Step {step_num} ({step_names[step_num]}): {step_times[step_num]:.1f}s ({step_times[step_num]/60:.1f} min)")

    logger.info(f"  Total: {total_elapsed:.1f}s ({total_elapsed/3600:.1f} hours)")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
