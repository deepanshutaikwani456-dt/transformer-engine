import json
from typing import List

class CharTokenizer:
    """Character-level tokenizer with encode, decode, and serialization support."""

    def __init__(self, chars: List[str] = None):
        if chars is not None:
            self.chars = sorted(list(set(chars)))
            self.vocab_size = len(self.chars)
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
            self.itos = {i: ch for i, ch in enumerate(self.chars)}
        else:
            self.chars = []
            self.vocab_size = 0
            self.stoi = {}
            self.itos = {}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(list(set(text)))
        return cls(chars)

    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of integer token IDs."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of integer token IDs back into a string."""
        return "".join([self.itos.get(i, "") for i in token_ids])

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["chars"])
