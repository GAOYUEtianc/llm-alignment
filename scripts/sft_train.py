#!/usr/bin/env python3
"""SFT Training Script for Qwen 2.5 Math 1.5B on GSM8K/MATH dataset.

Usage:
    python scripts/sft_train.py --num_examples 128 --learning_rate 1e-5

For multi-GPU setup (policy on GPU 0, vLLM on GPU 1):
    CUDA_VISIBLE_DEVICES=0,1 python scripts/sft_train.py --policy_device cuda:0 --vllm_device cuda:1
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import torch
from torch.optim import AdamW

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)


def load_r1_zero_prompt() -> str:
    """Load r1_zero prompt from file."""
    prompt_path = Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_gsm8k_dataset(dataset_path: str) -> list[dict]:
    """Load GSM8K dataset from JSONL file."""
    examples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_gsm8k_for_sft(
    examples: list[dict],
    prompt_template: str,
) -> list[dict]:
    """Format GSM8K examples for SFT training.

    Transforms {"question": str, "answer": str} to {"prompt": str, "response": str}
    where the response is wrapped in <think>...</think> <answer>...</answer> format.
    """
    formatted = []
    for example in examples:
        question = example.get("question") or example.get("problem")
        answer = example.get("answer") or example.get("solution")

        if question and answer:
            # Format prompt using r1_zero template
            prompt = prompt_template.format(question=question)

            # Extract final answer (after ####) for GSM8K format
            if "####" in answer:
                reasoning, final_answer = answer.rsplit("####", 1)
                reasoning = reasoning.strip()
                final_answer = final_answer.strip()
                # Format response with think/answer tags
                response = f"{reasoning}</think> <answer>\\boxed{{{final_answer}}}</answer>"
            else:
                # For other formats, use the whole answer
                response = f"{answer}</think> <answer>{answer}</answer>"

            formatted.append({
                "prompt": prompt,
                "response": response,
                "ground_truth": answer,
            })

    return formatted


def filter_correct_examples(
    examples: list[dict],
    model: LLM,
    prompt_template: str,
    reward_fn: Callable,
    sampling_params: SamplingParams,
) -> list[dict]:
    """Filter examples to only include those where the model produces correct answers."""
    print("Filtering dataset for correct examples...")

    # Generate responses for all examples
    prompts = [prompt_template.format(question=ex.get("question") or ex.get("problem"))
               for ex in examples]
    ground_truths = [ex.get("answer") or ex.get("solution") for ex in examples]

    outputs = model.generate(prompts, sampling_params)

    correct_examples = []
    for i, output in enumerate(tqdm(outputs, desc="Filtering")):
        response = output.outputs[0].text
        reward_info = reward_fn(response, ground_truths[i])
        if reward_info["reward"] > 0:
            correct_examples.append(examples[i])

    print(f"Filtered to {len(correct_examples)} correct examples out of {len(examples)}")
    return correct_examples


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    """Initialize vLLM instance on specified device."""
    vllm_set_random_seed(seed)

    # Monkeypatch from TRL
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """Load policy weights into vLLM instance."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_model(
    vllm_model: LLM,
    eval_prompts: list[str],
    eval_ground_truths: list[str],
    reward_fn: Callable,
    sampling_params: SamplingParams,
) -> dict:
    """Evaluate model using vLLM and return metrics."""
    outputs = vllm_model.generate(eval_prompts, sampling_params)

    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0
    correct_count = 0

    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        reward_info = reward_fn(response, eval_ground_truths[i])

        total_reward += reward_info["reward"]
        total_format_reward += reward_info["format_reward"]
        total_answer_reward += reward_info["answer_reward"]

        if reward_info["reward"] > 0:
            correct_count += 1

    n = len(eval_prompts)
    return {
        "accuracy": correct_count / n,
        "avg_reward": total_reward / n,
        "avg_format_reward": total_format_reward / n,
        "avg_answer_reward": total_answer_reward / n,
        "correct_count": correct_count,
        "total_count": n,
    }


def train_sft(
    policy: PreTrainedModel,
    tokenizer,
    train_examples: list[dict],
    eval_prompts: list[str],
    eval_ground_truths: list[str],
    vllm_model: LLM,
    reward_fn: Callable,
    args,
):
    """Main SFT training loop."""
    # Setup optimizer
    optimizer = AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Setup eval sampling params
    eval_sampling_params = SamplingParams(
        temperature=0.0,  # Greedy for evaluation
        max_tokens=args.max_gen_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Training loop
    num_examples = len(train_examples)
    total_steps = (num_examples * args.num_epochs) // args.batch_size

    print(f"\n{'='*60}")
    print(f"Starting SFT Training")
    print(f"{'='*60}")
    print(f"Number of training examples: {num_examples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Total training steps: {total_steps}")
    print(f"Eval every {args.eval_every} steps")
    print(f"{'='*60}\n")

    global_step = 0
    eval_step = 0

    # Initial evaluation
    print("Running initial evaluation...")
    load_policy_into_vllm_instance(policy, vllm_model)
    eval_metrics = evaluate_model(
        vllm_model, eval_prompts, eval_ground_truths, reward_fn, eval_sampling_params
    )
    print(f"Initial eval - Accuracy: {eval_metrics['accuracy']:.4f}")

    if args.use_wandb:
        wandb.log({
            "eval/accuracy": eval_metrics["accuracy"],
            "eval/avg_reward": eval_metrics["avg_reward"],
            "eval/avg_format_reward": eval_metrics["avg_format_reward"],
            "eval/avg_answer_reward": eval_metrics["avg_answer_reward"],
            "eval_step": eval_step,
        })
    eval_step += 1

    for epoch in range(args.num_epochs):
        # Shuffle training data
        random.shuffle(train_examples)

        epoch_loss = 0.0
        num_batches = 0

        policy.train()
        optimizer.zero_grad()

        # Create batches
        for batch_start in tqdm(
            range(0, num_examples, args.microbatch_size),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
        ):
            batch_end = min(batch_start + args.microbatch_size, num_examples)
            batch = train_examples[batch_start:batch_end]

            # Prepare batch data
            prompts = [ex["prompt"] for ex in batch]
            responses = [ex["response"] for ex in batch]

            # Tokenize
            tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids = tokenized["input_ids"].to(policy.device)
            labels = tokenized["labels"].to(policy.device)
            response_mask = tokenized["response_mask"].to(policy.device)

            # Get log probs
            log_prob_output = get_response_log_probs(
                policy, input_ids, labels, return_token_entropy=False
            )
            policy_log_probs = log_prob_output["log_probs"]

            # Compute loss and backward
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs,
                response_mask,
                args.gradient_accumulation_steps,
                normalize_constant=response_mask.sum().item(),
            )

            epoch_loss += loss.item()
            num_batches += 1

            # Gradient accumulation step
            if num_batches % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                avg_loss = epoch_loss / num_batches

                if args.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/epoch": epoch + 1,
                        "train_step": global_step,
                    })

                # Evaluation
                if global_step % args.eval_every == 0:
                    print(f"\nStep {global_step} - Running evaluation...")
                    policy.eval()
                    load_policy_into_vllm_instance(policy, vllm_model)

                    eval_metrics = evaluate_model(
                        vllm_model, eval_prompts, eval_ground_truths, reward_fn, eval_sampling_params
                    )

                    print(f"Step {global_step} - Accuracy: {eval_metrics['accuracy']:.4f}, "
                          f"Avg Loss: {avg_loss:.4f}")

                    if args.use_wandb:
                        wandb.log({
                            "eval/accuracy": eval_metrics["accuracy"],
                            "eval/avg_reward": eval_metrics["avg_reward"],
                            "eval/avg_format_reward": eval_metrics["avg_format_reward"],
                            "eval/avg_answer_reward": eval_metrics["avg_answer_reward"],
                            "eval_step": eval_step,
                        })
                    eval_step += 1

                    policy.train()

        # End of epoch evaluation
        print(f"\nEnd of Epoch {epoch + 1} - Running evaluation...")
        policy.eval()
        load_policy_into_vllm_instance(policy, vllm_model)

        eval_metrics = evaluate_model(
            vllm_model, eval_prompts, eval_ground_truths, reward_fn, eval_sampling_params
        )

        print(f"Epoch {epoch + 1} - Accuracy: {eval_metrics['accuracy']:.4f}, "
              f"Avg Loss: {epoch_loss / num_batches:.4f}")

        if args.use_wandb:
            wandb.log({
                "eval/accuracy": eval_metrics["accuracy"],
                "eval/avg_reward": eval_metrics["avg_reward"],
                "eval/avg_format_reward": eval_metrics["avg_format_reward"],
                "eval/avg_answer_reward": eval_metrics["avg_answer_reward"],
                "eval_step": eval_step,
            })
        eval_step += 1

    # Save final model
    if args.output_dir:
        output_path = Path(args.output_dir) / f"sft_model_n{len(train_examples)}"
        output_path.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"Model saved to {output_path}")

    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Qwen 2.5 Math")

    # Data arguments
    parser.add_argument("--train_data", type=str, default="data/gsm8k/train.jsonl",
                        help="Path to training data")
    parser.add_argument("--eval_data", type=str, default="data/gsm8k/test.jsonl",
                        help="Path to evaluation data")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Number of training examples (None for full dataset)")
    parser.add_argument("--num_eval_examples", type=int, default=500,
                        help="Number of evaluation examples")
    parser.add_argument("--filter_correct", action="store_true",
                        help="Filter training data to only correct examples")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                        help="Model name or path")
    parser.add_argument("--policy_device", type=str, default="cuda:0",
                        help="Device for policy model")
    parser.add_argument("--vllm_device", type=str, default="cuda:1",
                        help="Device for vLLM evaluation")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--microbatch_size", type=int, default=2,
                        help="Microbatch size for gradient accumulation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--max_gen_tokens", type=int, default=1024,
                        help="Maximum generation tokens for evaluation")

    # Evaluation arguments
    parser.add_argument("--eval_every", type=int, default=50,
                        help="Evaluate every N steps")

    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="sft-math",
                        help="Wandb project name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Wandb run name")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/sft",
                        help="Output directory for saved models")

    # Misc arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Adjust gradient accumulation
    args.gradient_accumulation_steps = args.batch_size // args.microbatch_size

    # Initialize wandb
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Disabling wandb logging.")
            args.use_wandb = False
        else:
            run_name = args.run_name or f"sft_n{args.num_examples or 'full'}_lr{args.learning_rate}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
            )
            # Setup wandb metrics
            wandb.define_metric("train_step")
            wandb.define_metric("eval_step")
            wandb.define_metric("train/*", step_metric="train_step")
            wandb.define_metric("eval/*", step_metric="eval_step")

    # Load prompt template
    print("Loading prompt template...")
    prompt_template = load_r1_zero_prompt()

    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.policy_device)

    # Initialize vLLM
    print(f"Initializing vLLM on {args.vllm_device}...")
    vllm_model = init_vllm(
        args.model_name,
        args.vllm_device,
        args.seed,
        gpu_memory_utilization=0.85,
    )

    # Load training data
    print(f"Loading training data from {args.train_data}")
    train_raw = load_gsm8k_dataset(args.train_data)

    # Filter for correct examples if requested
    if args.filter_correct:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_gen_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )
        train_raw = filter_correct_examples(
            train_raw, vllm_model, prompt_template, r1_zero_reward_fn, sampling_params
        )

    # Format training data
    train_examples = format_gsm8k_for_sft(train_raw, prompt_template)

    # Subset if specified
    if args.num_examples is not None and args.num_examples < len(train_examples):
        random.shuffle(train_examples)
        train_examples = train_examples[:args.num_examples]

    print(f"Training on {len(train_examples)} examples")

    # Load evaluation data
    print(f"Loading evaluation data from {args.eval_data}")
    eval_raw = load_gsm8k_dataset(args.eval_data)
    if args.num_eval_examples and args.num_eval_examples < len(eval_raw):
        eval_raw = eval_raw[:args.num_eval_examples]

    eval_prompts = [prompt_template.format(question=ex.get("question") or ex.get("problem"))
                    for ex in eval_raw]
    eval_ground_truths = [ex.get("answer") or ex.get("solution") for ex in eval_raw]

    print(f"Evaluating on {len(eval_prompts)} examples")

    # Train
    final_metrics = train_sft(
        policy=policy,
        tokenizer=tokenizer,
        train_examples=train_examples,
        eval_prompts=eval_prompts,
        eval_ground_truths=eval_ground_truths,
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        args=args,
    )

    print(f"\nFinal Accuracy: {final_metrics['accuracy']:.4f}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
