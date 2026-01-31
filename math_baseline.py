import json
import os
from pathlib import Path
from typing import List, Dict, Callable
from vllm import LLM, SamplingParams
from tqdm import tqdm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def load_r1_zero_prompt() -> str:
    """Load r1_zero prompt from file"""
    prompt_path = Path(__file__).parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Prompt file not found at {prompt_path}")
    
    
def load_dataset(dataset_path: str) -> List[Dict]:
    """Load dataset from JSONL file, support MATH & GSM8K formats"""
    examples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from {dataset_path}")
    return examples

def format_prompts(examples: List[Dict], prompt_template: str) -> tuple[List[str], List[str]]:
    """Format prompts and extract solutions from examples"""
    prompts = []
    solutions = []
    for example in examples:
        question = example.get("question") or example.get("problem")
        solution = example.get("solution") or example.get("answer")
        if question and solution:
            prompt = prompt_template.format(question=question)
            prompts.append(prompt)
            solutions.append(solution)
    return prompts, solutions

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truth_answers: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str=None
) -> Dict:
    """Use vLLM to evaluate model
    return a dictionary of evaluation results including statistics of evaluation
    """
    print(f"\nGenerating responses for {len(prompts)} prompts...")
    
    # use vLLM model to generate in batches 
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    # collect result and calculate rewards 
    results = []
    reward_stats = {
        'format_1_answer_1': 0, # format and and answer both correct
        'format_1_answer_0': 0, # format correct but answer wrong
        'format_0_answer_0': 0, # format and answer both wrong
    }
    
    total_format_reward = 0.0
    total_answer_reward = 0.0

    print("Evaluating responses...")
    for i, output in enumerate(tqdm(outputs)):
        response = output.outputs[0].text
        ground_truth = ground_truth_answers[i]

        # Calculate rewards using reward function
        rewards = reward_fn(response, ground_truth)

        # Track statistics
        format_reward = rewards['format_reward']
        answer_reward = rewards['answer_reward']

        total_format_reward += format_reward
        total_answer_reward += answer_reward

        # Categorize results
        if format_reward == 1.0 and answer_reward == 1.0:
            reward_stats['format_1_answer_1'] += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            reward_stats['format_1_answer_0'] += 1
        else:
            reward_stats['format_0_answer_0'] += 1

        # Store result
        results.append({
            'prompt': prompts[i],
            'response': response,
            'ground_truth': ground_truth,
            'format_reward': format_reward,
            'answer_reward': answer_reward,
            'reward': rewards['reward']
        })

    # Calculate aggregate metrics
    num_examples = len(prompts)
    avg_format_reward = total_format_reward / num_examples
    avg_answer_reward = total_answer_reward / num_examples

    eval_results = {
        'num_examples': num_examples,
        'avg_format_reward': avg_format_reward,
        'avg_answer_reward': avg_answer_reward,
        'reward_stats': reward_stats,
        'results': results
    }

    # Print summary
    print(f"\n{'='*60}")
    print("Evaluation Results Summary")
    print(f"{'='*60}")
    print(f"Total examples: {num_examples}")
    print(f"Average format reward: {avg_format_reward:.4f}")
    print(f"Average answer reward: {avg_answer_reward:.4f}")
    print(f"\nBreakdown:")
    print(f"  Format=1, Answer=1 (correct): {reward_stats['format_1_answer_1']} ({100*reward_stats['format_1_answer_1']/num_examples:.1f}%)")
    print(f"  Format=1, Answer=0 (wrong answer): {reward_stats['format_1_answer_0']} ({100*reward_stats['format_1_answer_0']/num_examples:.1f}%)")
    print(f"  Format=0, Answer=0 (format error): {reward_stats['format_0_answer_0']} ({100*reward_stats['format_0_answer_0']/num_examples:.1f}%)")
    print(f"{'='*60}\n")

    # Save results if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

    return eval_results


def main():
    """Main evaluation script for zero-shot MATH baseline"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Qwen 2.5 Math 1.5B on MATH dataset")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/Qwen2.5-Math-1.5B",
        help="Path or name of the model to evaluate"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/gsm8k/test.jsonl",
        help="Path to MATH validation dataset"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/math_baseline_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum generation length"
    )

    args = parser.parse_args()

    # Load prompt template
    print("Loading r1_zero prompt template...")
    prompt_template = load_r1_zero_prompt()

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    examples = load_dataset(args.dataset_path)

    if args.max_samples:
        examples = examples[:args.max_samples]
        print(f"Limited to {args.max_samples} samples for evaluation")

    # Format prompts
    print("Formatting prompts...")
    prompts, ground_truth_answers = format_prompts(examples, prompt_template)
    print(f"Prepared {len(prompts)} prompts")

    # Initialize vLLM model
    print(f"\nLoading model: {args.model_path}...")
    vllm_model = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9
    )

    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    print(f"\nSampling parameters:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Stop string: </answer>")

    # Run evaluation
    eval_results = evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truth_answers=ground_truth_answers,
        eval_sampling_params=sampling_params,
        output_path=args.output_path
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()