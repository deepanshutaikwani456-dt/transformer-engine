import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt import GPTLanguageModel, decode, encode, device, block_size

def load_trained_model(weights_path='model.pt'):
    """Load the trained GPT model from disk in 0.1 seconds!"""
    model = GPTLanguageModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def complete_prompt(model, prompt_text, max_new_tokens=250, temperature=1.0):
    """
    Takes a starting prompt string and generates the continuation.
    temperature:
      0.2 -> very focused, strict, repetitive
      0.8 -> natural, fluent Shakespeare
      1.5 -> wild, creative, chaotic
    """
    # 1. Turn prompt string into list of token IDs
    encoded_prompt = encode(prompt_text)
    # Shape: (1, T) - 1 sentence, T characters
    idx = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)

    print(f"\n=======================================================")
    print(f"PROMPT:      '{prompt_text}'")
    print(f"TEMPERATURE: {temperature}")
    print(f"=======================================================")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            # Focus only on the last token and apply TEMPERATURE
            logits = logits[:, -1, :] / temperature
            # Softmax to get percentages
            probs = F.softmax(logits, dim=-1)
            # Spin the wheel of fortune!
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

    generated_text = decode(idx[0].tolist())
    print(generated_text)
    print("-------------------------------------------------------\n")
    return generated_text

if __name__ == '__main__':
    model = load_trained_model('model.pt')

    # Example 1: Let Romeo speak at normal temperature (0.8)
    complete_prompt(model, prompt_text="ROMEO:\nO, wilt thou leave me", max_new_tokens=200, temperature=0.8)

    # Example 2: The King speaks with low temperature (0.3 - very focused)
    complete_prompt(model, prompt_text="KING RICHARD:\nWhat news, my Lord", max_new_tokens=200, temperature=0.3)

    # Example 3: Same King prompt with high temperature (1.4 - wild & creative)
    complete_prompt(model, prompt_text="KING RICHARD:\nWhat news, my Lord", max_new_tokens=200, temperature=1.4)
