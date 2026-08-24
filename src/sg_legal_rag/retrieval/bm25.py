from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .tokenization import tokenize


@dataclass(frozen=True)
class Ranking:
    top_indices: tuple[int, ...]
    first_relevant_rank: int | None
    positive_matches: int


class BM25Index:
    """A deterministic in-memory BM25 index using Lucene's positive IDF variant."""

    def __init__(self, documents: Sequence[str], *, k1: float = 1.2, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("BM25 requires at least one document")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b
        self.document_count = len(documents)
        self.term_frequencies: list[dict[str, int]] = []
        self.document_lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for index, document in enumerate(documents):
            frequencies = Counter(tokenize(document))
            self.term_frequencies.append(dict(frequencies))
            length = sum(frequencies.values())
            self.document_lengths.append(length)
            for term, frequency in frequencies.items():
                postings[term].append((index, frequency))

        self.average_document_length = sum(self.document_lengths) / self.document_count
        self.postings = dict(postings)
        self.idf = {
            term: math.log(
                1.0 + (self.document_count - len(term_postings) + 0.5) / (len(term_postings) + 0.5)
            )
            for term, term_postings in self.postings.items()
        }

    def _term_score(self, term: str, frequency: int, document_index: int) -> float:
        length = self.document_lengths[document_index]
        normalization = self.k1 * (1.0 - self.b + self.b * length / self.average_document_length)
        return self.idf[term] * (frequency * (self.k1 + 1.0)) / (frequency + normalization)

    def scores(
        self, query: str, candidate_indices: Iterable[int] | None = None
    ) -> dict[int, float]:
        query_terms = set(tokenize(query)) & self.idf.keys()
        if not query_terms:
            return {}

        scores: dict[int, float] = defaultdict(float)
        if candidate_indices is None:
            for term in query_terms:
                for document_index, frequency in self.postings[term]:
                    scores[document_index] += self._term_score(term, frequency, document_index)
            return dict(scores)

        for document_index in candidate_indices:
            if not 0 <= document_index < self.document_count:
                raise IndexError(f"candidate document index out of range: {document_index}")
            for term, frequency in self.term_frequencies[document_index].items():
                if term in query_terms:
                    scores[document_index] += self._term_score(term, frequency, document_index)
        return dict(scores)

    def rank(
        self,
        query: str,
        relevant_indices: set[int],
        *,
        top_k: int,
        candidate_indices: Sequence[int] | None = None,
    ) -> Ranking:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not relevant_indices:
            raise ValueError("at least one relevant document is required")

        if candidate_indices is not None:
            candidates = tuple(candidate_indices)
            candidate_set = set(candidates)
            if len(candidate_set) != len(candidates):
                raise ValueError("candidate indices must be unique")
            if not relevant_indices <= candidate_set:
                raise ValueError("relevant documents must occur in the candidate set")
            scores = self.scores(query, candidates)
            ranked = sorted(candidates, key=lambda index: (-scores.get(index, 0.0), index))
            first_relevant_rank = next(
                rank for rank, index in enumerate(ranked, start=1) if index in relevant_indices
            )
            return Ranking(
                top_indices=tuple(ranked[:top_k]),
                first_relevant_rank=first_relevant_rank,
                positive_matches=len(scores),
            )

        if not all(0 <= index < self.document_count for index in relevant_indices):
            raise IndexError("relevant document index out of range")
        scores = self.scores(query)
        ranked_positive = sorted(scores, key=lambda index: (-scores[index], index))
        top_indices = ranked_positive[:top_k]
        if len(top_indices) < top_k:
            positive_set = set(scores)
            for index in range(self.document_count):
                if index not in positive_set:
                    top_indices.append(index)
                    if len(top_indices) == top_k:
                        break

        first_positive_rank = next(
            (
                rank
                for rank, index in enumerate(ranked_positive, start=1)
                if index in relevant_indices
            ),
            None,
        )
        if first_positive_rank is not None:
            first_relevant_rank = first_positive_rank
        else:
            first_zero_relevant = min(relevant_indices)
            positive_before_or_at = sum(index <= first_zero_relevant for index in scores)
            zero_position = first_zero_relevant + 1 - positive_before_or_at
            first_relevant_rank = len(ranked_positive) + zero_position

        return Ranking(
            top_indices=tuple(top_indices),
            first_relevant_rank=first_relevant_rank,
            positive_matches=len(scores),
        )
