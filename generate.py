import argparse
import sys
import os
import torch
from src.model import GPTLanguageModel, GPTConfig
from src.tokenizer import CharTokenizer
from src.visualize import render_ascii_attention

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using a trained GPT model.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/shakespeare_model.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/shakespeare_tokenizer.json", help="Path to tokenizer JSON")
    parser.add_argument("--prompt", type=str, default="ROMEO:\nO, wilt thou leave me", help="Initial prompt text")
    parser.add_argument("--max_tokens", type=int, default=250, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-K filtering threshold")
    parser.add_argument("--top_p", type=float, default=None, help="Top-P (Nucleus) filtering threshold")
    parser.add_argument("--visualize_attention", action="store_true", help="Display attention heatmap matrix")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Tokenizer
    if not os.path.exists(args.tokenizer):
        # Fallback to deriving from shakespeare if tokenizer json is not found
        if os.path.exists("data/shakespeare.txt"):
            with open("data/shakespeare.txt", "r") as f:
                tokenizer = CharTokenizer.from_text(f.read())
        else:
            print(f"Error: Tokenizer file '{args.tokenizer}' not found.")
            sys.exit(1)
    else:
        tokenizer = CharTokenizer.load(args.tokenizer)

    # 2. Load Model & Config
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found! Run train.py first.")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "config" in ckpt and "state_dict" in ckpt:
        config = ckpt["config"]
        state_dict = ckpt["state_dict"]
    else:
        # Backward compatibility with direct state_dict
        config = GPTConfig(vocab_size=tokenizer.vocab_size)
        state_dict = ckpt

    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"\n=======================================================")
    print(f"PROMPT:      '{args.prompt}'")
    print(f"CONFIG:      Temp={args.temperature} | Top-K={args.top_k} | Top-P={args.top_p}")
    print(f"=======================================================")

    encoded = tokenizer.encode(args.prompt)
    if len(encoded) == 0:
        encoded = [0]
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

    # 3. Optional Attention Visualization on Prompt
    if args.visualize_attention:
        print("\n--- Visualizing Self-Attention (Layer 0, Head 0) ---")
        with torch.no_grad():
            idx_vis = idx[:, -min(idx.shape[1], 12):] # Show last 12 chars
            _ = model(idx_vis, save_attention=True)
            attn_weights = model.blocks[0].sa.heads[0].last_attention_weights[0]
            token_chars = [tokenizer.decode([i]) for i in idx_vis[0].tolist()]
            # Clean display for newlines and spaces
            display_chars = [c.replace("\n", "\\n").replace(" ", "␣") for c in token_chars]
            print(render_ascii_attention(display_chars, attn_weights))
            print("----------------------------------------------------\n")

    # 4. Autoregressive Generation
    with torch.no_grad():
        out_idx = model.generate(
            idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    generated_text = tokenizer.decode(out_idx[0].tolist())
    print(generated_text)
    print("-------------------------------------------------------\n")

if __name__ == "__main__":
    main()
