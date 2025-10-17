import json
import regex as re
from pathlib import Path
from collections.abc import Iterable


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.token2id = {v: k for k, v in vocab.items()}
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
        pattern = "|".join(re.escape(st) for st in self.special_tokens)
        chunks = re.split(f"({pattern})", text)
        print(chunks)
        for i in range(len(chunks)):
            if chunks[i] in self.special_tokens:
                chunks[i] = self.token2id[chunks[i].encode("utf-8")]
        chunks = [chunk for chunk in chunks if chunk != ""]
        print(chunks)

        PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        tokens = []
        for chunk in chunks:
            if isinstance(chunk, int):
                tokens.append(chunk)
                continue
            for match in re.finditer(PAT, chunk):
                word = match.group()
                pre_token = [byte for byte in word.encode("utf-8")]

                print("word:", word)
                print("pre_token:", pre_token)
                token = []
                change = 1
                while change:
                    change = 0
                    i = 0
                    while i < len(pre_token) - 1:
                        pair = (self.vocab[pre_token[i]], self.vocab[pre_token[i + 1]])
                        if pair in self.merges:
                            new_token = self.token2id[pair[0] + pair[1]]
                            pre_token = pre_token[:i] + [new_token] + pre_token[i + 2 :]
                            change = 1
                            continue
                        i += 1
                tokens.extend(pre_token)

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        pass


def path(file_name: str) -> str:
    return Path(__file__).resolve().parent / file_name


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(path("vocab.json"), path("merges.json"), ["<|endoftext|>"])
    print(tokenizer)
    # print(tokenizer.vocab)
    # print(tokenizer.merges)
    # print(tokenizer)
    print(tokenizer.encode("the cat ate<|endoftext|>"))
    # print(tokenizer.decode([1, 2, 3]))
