import argparse
import os
import sys
import torch
from src.model import GPTLanguageModel, GPTConfig
from src.tokenizer import CharTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPT model from scratch.")
    parser.add_argument("--data", type=str, default="data/shakespeare.txt", help="Path to text corpus")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/shakespeare_model.pt", help="Path to save weights")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/shakespeare_tokenizer.json", help="Path to save tokenizer")
    parser.add_argument("--iters", type=int, default=2000, help="Total training steps")
    parser.add_argument("--eval_interval", type=int, default=200, help="Evaluation interval")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (B)")
    parser.add_argument("--block_size", type=int, default=64, help="Context length (T)")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension (C)")
    parser.add_argument("--n_head", type=int, default=4, help="Attention heads")
    parser.add_argument("--n_layer", type=int, default=4, help="Transformer layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n==========================================")
    print(f"       GPT TRAINING ENGINE (FROM SCRATCH) ")
    print(f"==========================================")
    print(f"Compute Device:   {device}")
    print(f"Training Data:    {args.data}")
    print(f"Target Model:     {args.checkpoint}")
    print(f"Iterations:       {args.iters}")
    print(f"Batch Size (B):   {args.batch_size}")
    print(f"Block Size (T):   {args.block_size}")
    print(f"Embedding Dim:    {args.n_embd}")
    print(f"Heads / Layers:   {args.n_head} heads / {args.n_layer} layers")
    print(f"==========================================\n")

    if not os.path.exists(args.data):
        print(f"Error: dataset file '{args.data}' not found!")
        sys.exit(1)

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Initialize & save Tokenizer
    tokenizer = CharTokenizer.from_text(text)
    os.makedirs(os.path.dirname(args.tokenizer_path) or ".", exist_ok=True)
    tokenizer.save(args.tokenizer_path)
    print(f"Vocabulary Size:  {tokenizer.vocab_size} unique characters")
    print(f"Saved Tokenizer:  {args.tokenizer_path}\n")

    # 2. Encode dataset
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
        x = torch.stack([d[i : i + args.block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # 3. Model setup
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    model = GPTLanguageModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Trainable Parameters: {total_params:,}\n")

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(25)
            for k in range(25):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("--- Training Progress ---")
    for iter in range(args.iters):
        if iter % args.eval_interval == 0 or iter == args.iters - 1:
            losses = estimate_loss()
            print(f"step {iter:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 4. Save Checkpoint
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({"config": config, "state_dict": model.state_dict()}, args.checkpoint)
    print(f"\nTraining Complete! Checkpoint saved to: {args.checkpoint}\n")

if __name__ == "__main__":
    main()
