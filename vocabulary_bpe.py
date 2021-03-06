import json
import os
from functools import lru_cache
from typing import List

import numpy as np
import regex as re
import requests
from tqdm import tqdm

"""Byte pair encoding utilities"""
"""Adapted from https://github.com/openai/gpt-2"""


@lru_cache()
def bytes_to_unicode() -> dict:
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: tuple) -> set:
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(
        self, encoder: dict, bpe_merges: List[tuple], errors: str = "replace"
    ) -> None:
        self.map_encode = encoder
        self.map_decode = {v: k for k, v in self.map_encode.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    def bpe(self, token: str):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.map_encode[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def encode_file(self, path: str, disable_tqdm: bool = False) -> np.ndarray:
        with open(path) as file:
            return np.concatenate(
                [self.encode(line) for line in tqdm(file, disable=disable_tqdm)]
            )

    def decode(self, tokens: List[int]) -> str:
        text = "".join([self.map_decode[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode(
            "utf-8", errors=self.errors
        )
        return text

    def __len__(self) -> int:
        return len(self.map_encode)


def get_encoder(model_name: str = "117M") -> Encoder:
    maybe_download_vocab(model_name)
    with open("encoder.json", "r") as f:
        encoder = json.load(f)
    with open("vocab.bpe", "r", encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


def maybe_download_vocab(model_name: str = "117M") -> None:
    url = "https://storage.googleapis.com/gpt-2/models/{}/".format(model_name)

    for filename in [
        # "checkpoint",
        "encoder.json",
        # "hparams.json",
        # "model.ckpt.data-00000-of-00001",
        # "model.ckpt.index",
        # "model.ckpt.meta",
        "vocab.bpe",
    ]:
        if not os.path.isfile(filename):
            with open(filename, "wb") as file:
                file.write(requests.get(url + filename).content)
