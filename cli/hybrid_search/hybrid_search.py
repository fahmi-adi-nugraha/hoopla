import json
from pathlib import Path
from typing import Any

from keyword_search.inverted_index import InvertedIndex
from scipy.stats import sem
from semantic_search.semantic_search import ChunkedSemanticSearch

DATA_DIR = "data"
HYBRID_ALPHA = 0.5
HYBRID_LIMIT = 5


def load_documents() -> list[dict[str, Any]]:
    with open(Path(DATA_DIR, "movies.json"), "r") as mfiles:
        return json.load(mfiles)["movies"]


class HybridSearch:
    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx: InvertedIndex = InvertedIndex()
        if not self.idx.index_path.exists():
            self.idx.build()
        else:
            self.idx.load()

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
        # return list(zip(doc_ids, normed_scores))
        return dict(zip(doc_ids, normed_scores))

    def hybrid_score(
        self, bm25_score: float, semantic_score: float, alpha: float = HYBRID_ALPHA
    ) -> float:
        return alpha * bm25_score + (1 - alpha) * semantic_score

    def weighted_search(
        self, query: str, alpha: float = HYBRID_ALPHA, limit=HYBRID_LIMIT
    ):
        new_limit = limit * 500

        bm25_scores = self._bm25_search(query, new_limit)
        bm25_scores_normed = self._normalize_with_doc_id(bm25_scores)

        cosine_scores = [
            (doc["id"], doc["score"])
            for doc in self.semantic_search.search_chunks(query, new_limit)
        ]
        cosine_scores_normed = self._normalize_with_doc_id(cosine_scores)

        final_scores = []
        doc_ids = set(
            list(bm25_scores_normed.keys()) + list(cosine_scores_normed.keys())
        )
        for doc_id in doc_ids:
            doc = self.idx.docmap[doc_id]
            bm25_score = bm25_scores_normed.get(doc_id, 0)
            semantic_score = cosine_scores_normed.get(doc_id, 0)
            final_scores.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"][:100],
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

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
