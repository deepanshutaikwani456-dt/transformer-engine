import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Hyperparameters (Optimized for fast local training on Mac)
# -----------------------------------------------------------------------------
batch_size = 32      # How many independent sequences do we process in parallel? (B)
block_size = 64      # Maximum context length for predictions (T) - was only 8 in Bigram!
max_iters = 2000     # Training iterations
eval_interval = 200  # How often to evaluate loss
learning_rate = 1e-3 # Transformer learning rate
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 25      # Fast evaluation batches
n_embd = 128         # Number of features describing each word (C)
n_head = 4           # Number of parallel attention heads (each head has 128 / 4 = 32 features)
n_layer = 4          # Number of Transformer blocks stacked (assembly line depth)
dropout = 0.2        # 20% of neurons randomly dropped during training
# -----------------------------------------------------------------------------

torch.manual_seed(1337)

# 1. Read Tiny Shakespeare dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Vocabulary & Tokenizer (Module 1)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 3. Train / Validation split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 4. Data Batching (The "Shift by 1" trick)
def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Module 3: Self-Attention (Communication Phase)
# -----------------------------------------------------------------------------
class Head(nn.Module):
    """One single head of self-attention (Q, K, V)."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Static triangular mask for causal masking (No Cheating!)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size) - Nametag
        q = self.query(x) # (B, T, head_size) - Question

        # Compute match scores ("affinities") scaled by 1/sqrt(head_size)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B, T, T)
        # Apply Causal Mask: blindfold future words with -infinity
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # Softmax turns scores into percentages (Module 3)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # Collect information (Values)
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from all heads side-by-side
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# -----------------------------------------------------------------------------
# Module 4: The Feed-Forward Layer (Thinking Phase)
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity (sitting alone at desk)."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 1. Expand scratchpad 4x
            nn.ReLU(),                     # 2. Filter (erase useless ideas)
            nn.Linear(4 * n_embd, n_embd), # 3. Compress back to summary
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# Module 5: The Transformer Block (The Assembly Line)
# -----------------------------------------------------------------------------
class Block(nn.Module):
    """One complete Transformer block: Talk (Attention) -> Think (Feed-Forward)"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # The Calmer (Module 5)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual Connections (The Highways): x + ...
        x = x + self.sa(self.ln1(x))   # Communication with Highway
        x = x + self.ffwd(self.ln2(x)) # Thinking with Highway
        return x


# -----------------------------------------------------------------------------
# Full GPT Model (Connecting Modules 1 through 6)
# -----------------------------------------------------------------------------
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Module 1: Word meaning lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Module 2: Seat number lookup table
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Module 5: Stack of Transformer blocks (The Assembly Line)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)
        # Module 6: Final Ballot (Logits projector to vocabulary)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 1. Look up word meanings and seat positions, then ADD them!
        tok_emb = self.token_embedding_table(idx)                         # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb                                            # (B, T, C)

        # 2. Run through all Transformer blocks (Talk -> Think -> Talk -> Think...)
        x = self.blocks(x)                                               # (B, T, C)
        x = self.ln_f(x)                                                 # (B, T, C)

        # 3. Cast raw votes for every word in the dictionary (Logits)
        logits = self.lm_head(x)                                         # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Flatten to 2D for cross-entropy penalty (Module 7)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) tensor of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size (prevent running out of seats!)
            idx_cond = idx[:, -block_size:]
            # Forward the model to get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the very last time step
            logits = logits[:, -1, :] # (B, C)
            # Apply softmax to get percentages
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Spin the wheel of fortune (Module 6)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append new token to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"Using compute device: {device}")
    model = GPTLanguageModel().to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,} (10.7 Million dials!)\n")

    # Optimizer (AdamW with weight decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("--- STARTING GPT TRAINING ---")
    for iter in range(max_iters):

        # Evaluate loss periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = get_batch('train')

        # Forward pass & Backpropagation
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True) # Wipe the chalkboard
        loss.backward()                       # Calculate blame for 10M dials
        optimizer.step()                      # Nudge the dials

    # Save the trained model weights to disk!
    torch.save(model.state_dict(), 'model.pt')
    print("Model weights successfully saved to 'model.pt'!\n")

    print("\n--- TEXT GENERATION FROM TRAINED GPT ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    print(generated_text)
    print("----------------------------------------\n")
