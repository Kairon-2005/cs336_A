from __future__ import annotations

import regex as re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import os
from multiprocessing import Pool
from typing import BinaryIO


# GPT-2 pre-tokenization regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

Bytes = bytes
Token = Tuple[Bytes, ...]
Pair = Tuple[Bytes, Bytes]

def _split_on_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    if not special_tokens:
        return [text]
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    return re.split(pattern, text)

def _pretokenize(text: str) -> List[str]:
    return [m.group(0) for m in re.finditer(PAT, text)]

def _bytes_of(token: str) -> Token:
    b = token.encode("utf-8")
    return tuple(bytes([x]) for x in b)

def _get_pairs(token: Token) -> List[Pair]:
    return [(token[i], token[i+1]) for i in range(len(token) - 1)]

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes)

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def _pretokenize_chunk(args) -> Counter[Token]:
    chunk_bytes, special_tokens = args
    text = chunk_bytes.decode("utf-8", errors="ignore")

    counter: Counter[Token] = Counter()

    for chunk in _split_on_special_tokens(text, special_tokens):
        if not chunk:
            continue
        for tok in _pretokenize(chunk):
            counter[_bytes_of(tok)] += 1

    return counter

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str],
) -> Tuple[Dict[int, Bytes], List[Pair]]:
    vocab: Dict[int, Bytes] = {}
    next_id = 0

    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    pretoken_counts : Counter[Token] = Counter()

    num_workers = os.cpu_count() or 4
    special_token_bytes = special_tokens[0].encode("utf-8") if special_tokens else b""
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, special_token_bytes)
        chunks = []
        for start, end in zip(boundaries[:-1],boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start))
    with Pool(processes=num_workers) as pool:
        results = pool.map(_pretokenize_chunk, [(chunk, special_tokens) for chunk in chunks],)
    for counter in results:
        pretoken_counts.update(counter)
        
    pair_counts: Counter[Pair] = Counter()
    token_pairs: Dict[Token, Counter[Pair]] = {}

    for token, freq in pretoken_counts.items():
        pairs = Counter(_get_pairs(token))
        token_pairs[token] = pairs
        for p, c in pairs.items():
            pair_counts[p] += c * freq
    
    merges: List[Pair] = []

    while len(vocab) < vocab_size and pair_counts:
        best_pair = max(pair_counts.items(), key=lambda x: (x[1],x[0]))[0]
        A, B = best_pair
        AB = A + B
        merges.append(best_pair)
        vocab[next_id] = AB
        next_id += 1

        new_pretoken_counts: Counter[Token] = Counter()
        new_token_pairs : Dict[Token, Counter[Pair]] = {}

        for token, freq in pretoken_counts.items():
            if best_pair not in token_pairs[token]:
                new_pretoken_counts[token] += freq
                new_token_pairs[token] = token_pairs[token]
                continue
                
            merged: List[Bytes] = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and token[i] == A and token[i + 1] == B:
                    merged.append(AB)
                    i += 2
                else:
                    merged.append(token[i])
                    i += 1

            merged_token = tuple(merged)
            new_pretoken_counts[merged_token] += freq
            pairs = Counter(_get_pairs(merged_token))
            new_token_pairs[merged_token] = pairs
        
        pair_counts: Counter[Pair] = Counter()
        for token, freq in new_pretoken_counts.items():
            for p, c in new_token_pairs[token].items():
                pair_counts[p] += c * freq
        
        pretoken_counts = new_pretoken_counts
        token_pairs = new_token_pairs

    return vocab, merges

