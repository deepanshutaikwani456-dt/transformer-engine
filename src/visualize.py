from typing import List
import torch

def render_ascii_attention(
    tokens: List[str],
    weights: torch.Tensor,
    threshold: float = 0.1,
) -> str:
    """
    Renders an ASCII heatmap grid of attention weights between tokens.
    weights shape: (T, T)
    """
    T = len(tokens)
    out = []
    out.append("       " + "".join([f"{t[:4]:>5}" for t in tokens]))
    out.append("       " + "━" * (T * 5))

    # Intensity levels
    levels = [" ", "░", "▒", "▓", "█"]

    for i in range(T):
        row_str = f"{tokens[i][:4]:>5} ┃"
        for j in range(T):
            w = weights[i, j].item()
            if w <= 0.001:
                char = "·"
            else:
                idx = min(int(w * len(levels)), len(levels) - 1)
                char = levels[idx]
            row_str += f"{char:>5}"
        out.append(row_str)

    out.append("\nLegend: [·: 0%] [░: 20%] [▒: 40%] [▓: 60%] [█: 80-100%]")
    return "\n".join(out)
