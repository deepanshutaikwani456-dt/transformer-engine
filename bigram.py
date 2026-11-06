import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
batch_size = 32      # How many independent sequences do we process in parallel? (B)
block_size = 8       # What is the maximum context length for predictions? (T)
max_iters = 3000     # How many training iterations to run
eval_interval = 300  # How often to check the validation loss
learning_rate = 1e-2 # Step size for the optimizer (0.01)
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
# -----------------------------------------------------------------------------

torch.manual_seed(1337)

# 1. Read the Tiny Shakespeare dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Extract all unique characters (The Dictionary / Vocabulary)
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 3. Create Tokenizer: Character <-> ID mapping
stoi = { ch: i for i, ch in enumerate(chars) } # string to integer
itos = { i: ch for i, ch in enumerate(chars) } # integer to string
encode = lambda s: [stoi[c] for c in s]        # takes a string, outputs list of integers
decode = lambda l: ''.join([itos[i] for i in l])# takes a list of integers, outputs string

# 4. Train / Validation split (90% training, 10% validation)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 5. Data loader: Grab a small batch of inputs (x) and targets (y)
def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    # Pick random starting spots in the text
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    # Grab the chunks (Input x, and Target y shifted by 1 letter!)
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

# 6. The Bigram Language Model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx) # (B, T, C) where C = vocab_size

        if targets is None:
            loss = None
        else:
            # Flatten to 2D for cross_entropy: (B*T, C) and (B*T)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self(idx)
            # Focus only on the last time step (B, C)
            logits = logits[:, -1, :] 
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution (Spin the wheel of fortune!)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"Using compute device: {device}")
    print(f"Total characters in dataset: {len(text):,}")
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Unique characters: {''.join(chars)}\n")

    model = BigramLanguageModel(vocab_size).to(device)

    # Test generation BEFORE training (expect pure gibberish)
    print("--- TEXT GENERATION BEFORE TRAINING (Untrained Random Model) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with token 0 ('\n')
    print(decode(model.generate(context, max_new_tokens=150)[0].tolist()))
    print("----------------------------------------------------------------\n")

    # Create the AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("--- TRAINING THE BIGRAM MODEL ---")
    for iter in range(max_iters):

        # Periodically evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = get_batch('train')

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True) # Wipe the chalkboard
        loss.backward()                       # Calculate blame (slopes)
        optimizer.step()                      # Nudge the dials

    print("\n--- TEXT GENERATION AFTER TRAINING (Trained Bigram Model) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))
    print("------------------------------------------------------------\n")
