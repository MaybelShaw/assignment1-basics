import json
from pathlib import Path
from collections.abc import Iterable


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[bytes, bytes], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_hex = json.load(vf)
        vocab = {int(k): bytes.fromhex(v) for k, v in vocab_hex.items()}

        with open(merges_filepath, "r", encoding="utf-8") as mf:
            merges_hex = json.load(mf)
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_hex]

        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        pass


def path(file_name: str) -> str:
    return Path(__file__).resolve().parent / file_name


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(path("vocab.json"), path("merges.json"))
    print(tokenizer)
    # print(tokenizer.vocab)
    # print(tokenizer.merges)
    # print(tokenizer)
    print(tokenizer.encode("Hello, world!"))
    # print(tokenizer.decode([1, 2, 3]))
