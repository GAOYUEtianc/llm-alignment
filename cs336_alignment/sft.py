"""Supervised Fine-Tuning (SFT) utilities for alignment."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    batch_size = len(prompt_strs)
    assert len(output_strs) == batch_size, "prompt_strs and output_strs must have the same length"

    # Tokenize prompts and outputs separately (without padding first)
    prompt_encodings = tokenizer(
        prompt_strs,
        add_special_tokens=True,
        padding=False,
        return_tensors=None,
    )

    output_encodings = tokenizer(
        output_strs,
        add_special_tokens=False,  # Don't add special tokens to outputs
        padding=False,
        return_tensors=None,
    )

    # Concatenate prompt and output token IDs for each example
    concatenated_input_ids = []
    prompt_lengths = []

    for i in range(batch_size):
        prompt_ids = prompt_encodings["input_ids"][i]
        output_ids = output_encodings["input_ids"][i]

        # Concatenate prompt and output
        full_ids = prompt_ids + output_ids
        concatenated_input_ids.append(full_ids)
        prompt_lengths.append(len(prompt_ids))

    # Find the maximum length
    max_length = max(len(ids) for ids in concatenated_input_ids)

    # Pad sequences and create masks
    padded_input_ids = []
    response_masks = []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # If no pad token is defined, use eos_token_id
        pad_token_id = tokenizer.eos_token_id

    for i in range(batch_size):
        ids = concatenated_input_ids[i]
        prompt_len = prompt_lengths[i]
        seq_len = len(ids)

        # Pad to max_length
        padding_length = max_length - seq_len
        padded_ids = ids + [pad_token_id] * padding_length

        # Create response mask: 1 for response tokens, 0 for prompt and padding
        mask = [0] * prompt_len + [1] * (seq_len - prompt_len) + [0] * padding_length

        padded_input_ids.append(padded_ids)
        response_masks.append(mask)

    # Convert to tensors
    padded_input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
    response_masks_tensor = torch.tensor(response_masks, dtype=torch.long)

    # Slice off the final token for input_ids
    # and slice off the first token for labels (shift right)
    input_ids = padded_input_ids_tensor[:, :-1]  # Remove last token
    labels = padded_input_ids_tensor[:, 1:]      # Remove first token (shifted)
    response_mask = response_masks_tensor[:, 1:]  # Shift mask to align with labels

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): the entropy for each next-token prediction.
    """
    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
    probs = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

    # Entropy: H(p) = -∑ p(x) * log(p(x))
    entropy = -(probs * log_probs).sum(dim=-1)  # (batch_size, seq_len)

    return entropy


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities from a causal language model,
    and optionally the entropy of the next token predictions.

    Args:
        model: torch.nn.Module, HuggingFace model used for scoring
            (placed on the correct device and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor of shape (batch_size, sequence_length),
            concatenated prompt + response tokens as produced by tokenization method.
        labels: torch.Tensor of shape (batch_size, sequence_length),
            labels as produced by tokenization method.
        return_token_entropy: bool, if True, also return per-token entropy by calling compute_entropy.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length),
                conditional log-probabilities log p_θ(x_t | x_{<t}).
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length),
                per-token entropy for each position (present only if return_token_entropy=True).
    """
    # Get logits from the model
    logits = model(input_ids).logits  # (batch_size, sequence_length, vocab_size)

    # Compute log probabilities using log_softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, sequence_length, vocab_size)

    # Gather the log probabilities corresponding to the labels
    # labels: (batch_size, sequence_length)
    # We need to select log_probs[i, j, labels[i, j]] for each i, j
    labels_expanded = labels.unsqueeze(-1)  # (batch_size, sequence_length, 1)
    selected_log_probs = torch.gather(log_probs, dim=-1, index=labels_expanded)
    selected_log_probs = selected_log_probs.squeeze(-1)  # (batch_size, sequence_length)

    result = {
        "log_probs": selected_log_probs,
    }

    # Optionally compute and return token entropy
    if return_token_entropy:
        token_entropy = compute_entropy(logits)  # (batch_size, sequence_length)
        result["token_entropy"] = token_entropy

    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float, the constant to divide by for normalization.
        dim: int | None, the dimension to sum along before normalization.
            If None, sum over all dimensions.

    Returns:
        torch.Tensor, the normalized sum, where masked elements (mask == 0) don't contribute to the sum.
    """
    # Apply mask: only keep elements where mask == 1
    # Convert mask to same dtype as tensor for proper multiplication
    masked_tensor = tensor * mask

    # Sum along the specified dimension (or all dimensions if dim is None)
    if dim is None:
        result = masked_tensor.sum()
    else:
        result = masked_tensor.sum(dim=dim)

    # Normalize by the constant
    result = result / normalize_constant

    return result


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch for SFT.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length),
            per-token log-probabilities from the SFT policy being trained.
        response_mask: torch.Tensor of shape (batch_size, sequence_length),
            1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps: int, number of microbatches per optimizer step.
        normalize_constant: float, the constant by which to divide the sum.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: scalar tensor, the microbatch loss (for logging, not adjusted for gradient accumulation).
            metadata: dict with metadata from the underlying loss call, and any other statistics.
    """
    # Compute negative log-likelihood (NLL) loss
    # SFT loss = -sum(log p(response tokens)) / normalize_constant
    # We want to maximize log probability, which is equivalent to minimizing negative log probability
    nll = -policy_log_probs  # (batch_size, sequence_length)

    # Use masked_normalize to sum over response tokens and normalize
    total_loss = masked_normalize(nll, response_mask, normalize_constant, dim=None)

    # Get batch size for averaging
    batch_size = policy_log_probs.shape[0]

    # Compute per-sample loss (average over batch)
    loss = total_loss / batch_size

    # Adjust for gradient accumulation and perform backward pass
    # Divide by gradient_accumulation_steps so that the effective gradient
    # is the average over all microbatches
    (loss / gradient_accumulation_steps).backward()

    # Return the per-sample loss for logging
    metadata = {}

    return loss, metadata


def log_generations(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    generation_kwargs: dict | None = None,
) -> dict[str, list | float]:
    """Log generations from a model for a set of prompts.

    This function generates responses from the model for given prompts and logs
    various metrics including rewards, entropy, and response lengths.

    Args:
        model: torch.nn.Module, the model to generate from.
        tokenizer: PreTrainedTokenizerBase, the tokenizer to use.
        prompts: list[str], the input prompts to generate responses for.
        ground_truths: list[str], the ground-truth answers for each prompt.
        reward_fn: Callable[[str, str], dict[str, float]], a function that takes
            (response, ground_truth) and returns a dict with keys:
            "format_reward", "answer_reward", and "reward".
        generation_kwargs: dict | None, optional kwargs to pass to model.generate().

    Returns:
        dict[str, list | float]: A dictionary containing:
            "prompts": list[str], the input prompts.
            "responses": list[str], the generated responses.
            "ground_truths": list[str], the ground-truth answers.
            "format_rewards": list[float], format reward for each response.
            "answer_rewards": list[float], answer reward for each response.
            "rewards": list[float], total reward for each response.
            "avg_entropy": float, average token entropy across all responses.
            "avg_response_length": float, average response length in tokens.
            "avg_response_length_correct": float, average response length for correct responses.
            "avg_response_length_incorrect": float, average response length for incorrect responses.
    """
    if generation_kwargs is None:
        generation_kwargs = {}

    # Set default generation parameters if not provided
    if "max_new_tokens" not in generation_kwargs:
        generation_kwargs["max_new_tokens"] = 512
    if "do_sample" not in generation_kwargs:
        generation_kwargs["do_sample"] = False
    if "pad_token_id" not in generation_kwargs:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Tokenize prompts
    prompt_encodings = tokenizer(
        prompts,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )

    # Move to model device
    device = next(model.parameters()).device
    input_ids = prompt_encodings["input_ids"].to(device)
    attention_mask = prompt_encodings["attention_mask"].to(device)

    # Store prompt lengths for extracting responses later
    prompt_lengths = attention_mask.sum(dim=1).tolist()

    # Generate responses
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=True,
            **generation_kwargs,
        )

    generated_ids = outputs.sequences
    scores = outputs.scores  # tuple of (batch_size, vocab_size) for each generated token

    # Decode responses (excluding the prompt)
    responses = []
    response_lengths = []
    for i, (gen_ids, prompt_len) in enumerate(zip(generated_ids, prompt_lengths)):
        # Extract only the generated tokens (after the prompt)
        response_ids = gen_ids[prompt_len:]
        # Remove padding tokens if present
        if tokenizer.pad_token_id is not None:
            response_ids = response_ids[response_ids != tokenizer.pad_token_id]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        responses.append(response_text)
        response_lengths.append(len(response_ids))

    # Compute rewards for each response
    format_rewards = []
    answer_rewards = []
    rewards = []
    for response, gt in zip(responses, ground_truths):
        reward_info = reward_fn(response, gt)
        format_rewards.append(reward_info["format_reward"])
        answer_rewards.append(reward_info["answer_reward"])
        rewards.append(reward_info["reward"])

    # Compute average token entropy from generation scores
    if len(scores) > 0:
        # Stack scores: (num_generated_tokens, batch_size, vocab_size)
        stacked_scores = torch.stack(scores, dim=0)
        # Transpose to (batch_size, num_generated_tokens, vocab_size)
        stacked_scores = stacked_scores.transpose(0, 1)
        # Compute entropy for each position
        token_entropies = compute_entropy(stacked_scores)  # (batch_size, num_generated_tokens)

        # Compute average entropy, weighted by actual response lengths
        total_entropy = 0.0
        total_tokens = 0
        for i, resp_len in enumerate(response_lengths):
            if resp_len > 0:
                # Only consider entropy for actual generated tokens
                actual_len = min(resp_len, token_entropies.shape[1])
                total_entropy += token_entropies[i, :actual_len].sum().item()
                total_tokens += actual_len
        avg_entropy = total_entropy / total_tokens if total_tokens > 0 else 0.0
    else:
        avg_entropy = 0.0

    # Compute average response lengths
    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0.0

    # Separate correct and incorrect responses based on reward
    correct_lengths = [length for length, reward in zip(response_lengths, rewards) if reward > 0]
    incorrect_lengths = [length for length, reward in zip(response_lengths, rewards) if reward == 0]

    avg_response_length_correct = sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0.0
    avg_response_length_incorrect = sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else 0.0

    return {
        "prompts": prompts,
        "responses": responses,
        "ground_truths": ground_truths,
        "format_rewards": format_rewards,
        "answer_rewards": answer_rewards,
        "rewards": rewards,
        "avg_entropy": avg_entropy,
        "avg_response_length": avg_response_length,
        "avg_response_length_correct": avg_response_length_correct,
        "avg_response_length_incorrect": avg_response_length_incorrect,
    }