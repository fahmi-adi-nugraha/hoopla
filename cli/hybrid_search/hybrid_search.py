import json
from pathlib import Path
from typing import Any

from keyword_search.inverted_index import InvertedIndex
from semantic_search.semantic_search import ChunkedSemanticSearch

DATA_DIR = "data"
WEIGHTED_ALPHA = 0.5
RRF_K = 60
HYBRID_LIMIT = 5


class HybridSearch:
    def __init__(self, documents: list[dict[str, Any]] | None = None):
        if documents is None:
            self.documents = self._load_documents()
        else:
            self.documents = documents

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(self.documents)

        self.idx: InvertedIndex = InvertedIndex()
        if not self.idx.index_path.exists():
            self.idx.build()
        else:
            self.idx.load()

    def _load_documents(self) -> list[dict[str, Any]]:
        with open(Path(DATA_DIR, "movies.json"), "r") as mfiles:
            return json.load(mfiles)["movies"]

    def _bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def normalize(self, scores: list[int | float]) -> list[float]:
        if len(scores) == 0:
            return []

        min_score, max_score = min(scores), max(scores)
        if min_score == max_score:
            return [1.0 for _ in scores]

        scores_normed: list[float] = []
        for score in scores:
            scores_normed.append((score - min_score) / (max_score - min_score))
        return scores_normed

    def _normalize_with_doc_id(
        self, score_map: list[tuple[int, float]]
    ) -> dict[int, float]:
        doc_ids, scores = zip(*score_map)
        normed_scores = self.normalize(scores)
        return dict(zip(doc_ids, normed_scores))

    def hybrid_score(
        self, bm25_score: float, semantic_score: float, alpha: float = WEIGHTED_ALPHA
    ) -> float:
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def weighted_search(
        self, query: str, alpha: float = WEIGHTED_ALPHA, limit=HYBRID_LIMIT
    ):
        new_limit = limit * 500

        bm25_scores = self._bm25_search(query, new_limit)
        bm25_scores_normed = self._normalize_with_doc_id(bm25_scores)

        semantic_scores = [
            (doc["id"], doc["score"])
            for doc in self.semantic_search.search_chunks(query, new_limit)
        ]
        semantic_scores_normed = self._normalize_with_doc_id(semantic_scores)

        final_scores = []
        doc_ids = set(
            list(bm25_scores_normed.keys()) + list(semantic_scores_normed.keys())
        )
        for doc_id in doc_ids:
            doc = self.idx.docmap[doc_id]
            bm25_score = bm25_scores_normed.get(doc_id, 0)
            semantic_score = semantic_scores_normed.get(doc_id, 0)
            final_scores.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "hybrid_score": self.hybrid_score(
                        bm25_score, semantic_score, alpha
                    ),
                    "bm25_score": bm25_score,
                    "semantic_score": semantic_score,
                }
            )

        return sorted(final_scores, key=lambda x: x["hybrid_score"], reverse=True)[
            :limit
        ]

    def rrf_score(self, rank: int, k: int) -> float:
        return 1 / (k + rank)

    # We assume that the scores have already been sorted
    def _get_rrf_score_with_rank(
        self, scores: list[tuple[int, float]], k: int
    ) -> dict[int, tuple[float, int]]:
        rrf_scores = {}
        for i, (doc_id, score) in enumerate(scores):
            rank = i + 1
            rrf_scores[doc_id] = (self.rrf_score(rank, k), rank)
        return rrf_scores

    def rrf_search(
        self, query: str, k: int = RRF_K, limit: int = HYBRID_LIMIT
    ) -> list[dict[str, str | int | float]]:
        new_limit = limit * 500

        bm25_scores = self._bm25_search(query, new_limit)
        bm25_rrf = self._get_rrf_score_with_rank(bm25_scores, k)

        semantic_scores = [
            (doc["id"], doc["score"])
            for doc in self.semantic_search.search_chunks(query, new_limit)
        ]
        semantic_rrf = self._get_rrf_score_with_rank(semantic_scores, k)

        final_scores = []
        doc_ids = set(list(bm25_rrf.keys()) + list(semantic_rrf.keys()))
        for doc_id in doc_ids:
            doc = self.idx.docmap[doc_id]
            bm25_vals = bm25_rrf.get(doc_id)
            semantic_vals = semantic_rrf.get(doc_id)
            if bm25_vals is None or semantic_vals is None:
                continue
            bm25_score, bm25_rank = bm25_vals
            semantic_score, semantic_rank = semantic_vals
            final_scores.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "rrf_score": bm25_score + semantic_score,
                    "bm25_rank": bm25_rank,
                    "semantic_rank": semantic_rank,
                }
            )

        return sorted(final_scores, key=lambda x: x["rrf_score"], reverse=True)[:limit]
