from typing import Any

from keyword_search.inverted_index import InvertedIndex
from semantic_search.semantic_search import ChunkedSemanticSearch


def normalize(scores: list[int | float]) -> list[float]:
    if len(scores) == 0:
        return []

    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        return [1.0 for _ in scores]

    scores_normed: list[float] = []
    for score in scores:
        scores_normed.append((score - min_score) / (max_score - min_score))
    return scores_normed


class HybridSearch:
    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx: InvertedIndex = InvertedIndex()
        if not self.idx.index_path.exists():
            self.idx.build()

    def _bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
