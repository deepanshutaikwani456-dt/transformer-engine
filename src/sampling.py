import torch
import torch.nn.functional as F

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
) -> torch.Tensor:
    """
    Advanced autoregressive decoding with Temperature, Top-K, and Top-P (Nucleus) filtering.

    Args:
        logits: Raw prediction tensor of shape (batch_size, vocab_size).
        temperature: Controls randomness (lower = conservative, higher = creative).
                     If temperature <= 1e-4, greedy argmax selection is used.
        top_k: If set > 0, only sample from the top K most probable tokens.
        top_p: If set (0.0 < top_p < 1.0), sample from the smallest set of tokens
               whose cumulative probability exceeds top_p (Nucleus Sampling).
    """
    # 1. Greedy decoding if temperature is near 0
    if temperature <= 1e-4:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. Temperature scaling
    logits = logits / temperature

    # 3. Top-K filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Find the threshold value for top_k
        v, _ = torch.topk(logits, top_k)
        # Any logit smaller than the k-th logit gets masked out to -infinity
        logits[logits < v[:, [-1]]] = -float('Inf')

    # 4. Top-P (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Scatter sorted masks back to original logit indices
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('Inf')

    # 5. Softmax to get probability distribution
    probs = F.softmax(logits, dim=-1)

    # 6. Sample token index from distribution (Spin the wheel!)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
