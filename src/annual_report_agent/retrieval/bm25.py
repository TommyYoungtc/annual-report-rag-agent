from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

from ..schemas import Chunk, SearchResult
from .tokenization import tokenize


class BM25Retriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        if not chunks:
            raise ValueError("chunks cannot be empty")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self._term_frequencies: list[Counter[str]] = []
        document_frequency: defaultdict[str, int] = defaultdict(int)
        lengths: list[int] = []

        for chunk in self.chunks:
            terms = tokenizer(f"{chunk.company} {chunk.year} {chunk.section} {chunk.text}")
            frequencies = Counter(terms)
            self._term_frequencies.append(frequencies)
            lengths.append(len(terms))
            for term in frequencies:
                document_frequency[term] += 1

        self._lengths = lengths
        self._average_length = sum(lengths) / len(lengths)
        corpus_size = len(self.chunks)
        self._idf = {
            term: math.log(1 + (corpus_size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        allowed_document_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        query_terms = self.tokenizer(query)
        scored: list[tuple[int, float]] = []

        for index, frequencies in enumerate(self._term_frequencies):
            if (
                allowed_document_ids is not None
                and self.chunks[index].document_id not in allowed_document_ids
            ):
                continue
            score = 0.0
            document_length = self._lengths[index]
            length_norm = 1 - self.b + self.b * document_length / self._average_length
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * length_norm
                score += self._idf.get(term, 0.0) * numerator / denominator
            if score > 0:
                scored.append((index, score))

        scored.sort(key=lambda item: (-item[1], self.chunks[item[0]].chunk_id))
        return [
            SearchResult(
                chunk=self.chunks[index],
                score=score,
                rank=rank,
                source="bm25",
            )
            for rank, (index, score) in enumerate(scored[:top_k], start=1)
        ]
