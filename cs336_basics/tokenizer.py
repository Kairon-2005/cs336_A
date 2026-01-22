from __future__ import annotations

import pickle
import regex as re
from typing import Dict, Iterable, Iterator, List, Tuple

from cs336_basics.bpe import (
    PAT,
    _pretokenize,
    _bytes_of,
    Bytes,
    Token,
    Pair,
)

def split_keep_special(text: str, special_tokens: List[str]) -> List[str]:
        if not special_tokens:
            return [text]
        
        special_tokens = sorted(special_tokens, key=len, reverse=True)

        parts = []
        i = 0
        n = len(text)

        while i < n:
            matched = False
            for tok in special_tokens:
                if text.startswith(tok, i):
                    parts.append(tok)
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue

            parts.append(text[i])
            i += 1

        merged = []
        for p in parts:
            if merged and p not in special_tokens and merged[-1] not in special_tokens:
                merged[-1] += p
            else:
                merged.append(p)

        return merged
    
class Tokenizer:
    def __init__(self,
                 vocab: Dict[int, bytes],
                 merges: List[Pair],
                 special_tokens: List[str] | None = None,):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.bytes_to_id : Dict[bytes, int] = {b: i for i, b in vocab.items()}
        self.merge_ranks: Dict[Pair, int] = {pair: idx for idx, pair in enumerate(merges)}
        self.special_to_id: Dict[str, int] = {}
        for s in self.special_tokens:
            sb = s.encode("utf-8")
            if sb not in self.bytes_to_id:
                raise ValueError(f"Special token {s} not present in vocab")
            self.special_to_id[s] = self.bytes_to_id[sb]
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
            return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _apply_bpe(self, token: Token) -> List[bytes]:
        tokens: List[bytes] = list(token)
        if len(tokens) <= 1:
            return tokens
        
        while True:
            best_rank = None
            best_pair = None

            for i in range(len(tokens) - 1):
                p = (tokens[i], tokens[i + 1])
                r = self.merge_ranks.get(p)
                if r is None:
                    continue
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_pair = p

            if best_pair is None:
                break

            A, B = best_pair
            AB = A + B

            new_tokens: List[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == A and tokens[i + 1] == B:
                    new_tokens.append(AB)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

            if len(tokens) <= 1:
                break
        
        return tokens
    

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []

        for piece in split_keep_special(text, self.special_tokens):
            if piece in self.special_to_id:
                ids.append(self.special_to_id[piece])
                continue

            for pretoken in _pretokenize(piece):
                base = _bytes_of(pretoken)
                merged = self._apply_bpe(base)
                for b in merged:
                    try:
                        ids.append(self.bytes_to_id[b])
                    except KeyError as e:
                        raise KeyError(f"Token bytes not in vocab: {b!r}") from e
        
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            for tid in self.encode(s):
                yield tid
    
    def decode(self, ids: List[int]) -> str:
        byte_stream = b"".join(self.vocab[i] for i in ids)
        return byte_stream.decode("utf-8", errors="replace")