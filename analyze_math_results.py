#!/usr/bin/env python3
"""
Helper script to analyze MATH evaluation results.
This helps answer part (b) of the assignment.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import random


def load_results(results_path: str) -> Dict:
    """Load evaluation results from JSON file"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def categorize_results(results: List[Dict]) -> Dict:
    """Categorize results into three categories"""
    categories = {
        'format_1_answer_1': [],  # Correct format and answer
        'format_1_answer_0': [],  # Correct format, wrong answer
        'format_0_answer_0': []   # Format error (and wrong answer)
    }

    for result in results:
        format_reward = result['format_reward']
        answer_reward = result['answer_reward']

        if format_reward == 1.0 and answer_reward == 1.0:
            categories['format_1_answer_1'].append(result)
        elif format_reward == 1.0 and answer_reward == 0.0:
            categories['format_1_answer_0'].append(result)
        else:
            categories['format_0_answer_0'].append(result)

    return categories


def print_statistics(eval_results: Dict):
    """Print summary statistics"""
    print("="*80)
    print("MATH EVALUATION RESULTS SUMMARY")
    print("="*80)
    print(f"\nTotal examples evaluated: {eval_results['num_examples']}")
    print(f"Average format reward: {eval_results['avg_format_reward']:.4f}")
    print(f"Average answer reward: {eval_results['avg_answer_reward']:.4f}")
    print(f"\nAccuracy: {eval_results['avg_answer_reward']*100:.2f}%")

    stats = eval_results['reward_stats']
    total = eval_results['num_examples']

    print(f"\n{'Category':<40} {'Count':<10} {'Percentage':<10}")
    print("-"*80)
    print(f"{'Format=1, Answer=1 (Fully Correct)':<40} {stats['format_1_answer_1']:<10} {100*stats['format_1_answer_1']/total:.1f}%")
    print(f"{'Format=1, Answer=0 (Wrong Answer)':<40} {stats['format_1_answer_0']:<10} {100*stats['format_1_answer_0']/total:.1f}%")
    print(f"{'Format=0, Answer=0 (Format Error)':<40} {stats['format_0_answer_0']:<10} {100*stats['format_0_answer_0']/total:.1f}%")
    print("="*80)


def print_examples(categories: Dict, num_examples: int = 10):
    """Print example responses from each category"""

    # Format errors (format_0_answer_0)
    print("\n" + "="*80)
    print(f"CATEGORY 1: FORMAT ERRORS (Format=0, Answer=0)")
    print(f"Total: {len(categories['format_0_answer_0'])}")
    print("="*80)

    format_errors = categories['format_0_answer_0'][:num_examples]
    for i, result in enumerate(format_errors, 1):
        print(f"\n--- Example {i} ---")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Model Response:")
        print(result['response'][:500])  # Print first 500 chars
        if len(result['response']) > 500:
            print("... (truncated)")
        print()

    # Wrong answers with correct format (format_1_answer_0)
    print("\n" + "="*80)
    print(f"CATEGORY 2: CORRECT FORMAT, WRONG ANSWER (Format=1, Answer=0)")
    print(f"Total: {len(categories['format_1_answer_0'])}")
    print("="*80)

    wrong_answers = categories['format_1_answer_0'][:num_examples]
    for i, result in enumerate(wrong_answers, 1):
        print(f"\n--- Example {i} ---")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Model Response:")
        print(result['response'][:500])
        if len(result['response']) > 500:
            print("... (truncated)")
        print()

    # Fully correct (format_1_answer_1)
    print("\n" + "="*80)
    print(f"CATEGORY 3: FULLY CORRECT (Format=1, Answer=1)")
    print(f"Total: {len(categories['format_1_answer_1'])}")
    print("="*80)

    correct = categories['format_1_answer_1'][:min(3, len(categories['format_1_answer_1']))]
    for i, result in enumerate(correct, 1):
        print(f"\n--- Example {i} ---")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Model Response:")
        print(result['response'][:500])
        if len(result['response']) > 500:
            print("... (truncated)")
        print()


def analyze_format_errors(format_errors: List[Dict], num_examples: int = 10):
    """Detailed analysis of format errors"""
    print("\n" + "="*80)
    print("DETAILED ANALYSIS: FORMAT ERRORS")
    print("="*80)

    # Check for common issues
    missing_think_close = 0
    missing_answer_open = 0
    missing_answer_close = 0
    other_issues = 0

    for result in format_errors:
        response = result['response']
        has_think_close = '</think>' in response
        has_answer_open = '<answer>' in response
        has_answer_close = '</answer>' in response

        if not has_think_close:
            missing_think_close += 1
        elif not has_answer_open:
            missing_answer_open += 1
        elif not has_answer_close:
            missing_answer_close += 1
        else:
            other_issues += 1

    total = len(format_errors)
    print(f"\nFormat Error Breakdown (out of {total} format errors):")
    print(f"  Missing </think> tag: {missing_think_close} ({100*missing_think_close/total:.1f}%)")
    print(f"  Missing <answer> tag: {missing_answer_open} ({100*missing_answer_open/total:.1f}%)")
    print(f"  Missing </answer> tag: {missing_answer_close} ({100*missing_answer_close/total:.1f}%)")
    print(f"  Other issues: {other_issues} ({100*other_issues/total:.1f}%)")

    print(f"\nShowing {num_examples} random format error examples:\n")

    sample = random.sample(format_errors, min(num_examples, len(format_errors)))
    for i, result in enumerate(sample, 1):
        response = result['response']
        print(f"--- Format Error Example {i} ---")
        print(f"Ground truth: {result['ground_truth']}")
        print(f"Response (first 300 chars): {response[:300]}")
        if len(response) > 300:
            print("...")
        print(f"Response (last 100 chars): ...{response[-100:]}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MATH evaluation results"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="outputs/math_baseline_results.json",
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples to show per category"
    )
    parser.add_argument(
        "--detailed-format-analysis",
        action="store_true",
        help="Perform detailed analysis of format errors"
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_path}...")
    eval_results = load_results(args.results_path)

    # Print statistics
    print_statistics(eval_results)

    # Categorize results
    categories = categorize_results(eval_results['results'])

    # Print examples
    print_examples(categories, num_examples=args.num_examples)

    # Detailed format error analysis
    if args.detailed_format_analysis and len(categories['format_0_answer_0']) > 0:
        analyze_format_errors(categories['format_0_answer_0'], num_examples=args.num_examples)

    # Save categorized results for further analysis
    output_path = Path(args.results_path).parent / "categorized_results.json"
    categorized_summary = {
        'format_1_answer_1': {
            'count': len(categories['format_1_answer_1']),
            'examples': categories['format_1_answer_1'][:5]  # Save first 5 of each
        },
        'format_1_answer_0': {
            'count': len(categories['format_1_answer_0']),
            'examples': categories['format_1_answer_0'][:5]
        },
        'format_0_answer_0': {
            'count': len(categories['format_0_answer_0']),
            'examples': categories['format_0_answer_0'][:5]
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(categorized_summary, f, indent=2, ensure_ascii=False)

    print(f"\nCategorized summary saved to {output_path}")


if __name__ == "__main__":
    main()
