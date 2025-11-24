import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CACHE_DIR = "cache"
MOVIES_FILE_PATH = "data/movies.json"


def verify_model() -> None:
    sem_search = SemanticSearch()
    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")


def embed_text(text: str) -> None:
    sem_search = SemanticSearch()
    embedding: NDArray[np.float64] = sem_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    sem_search = SemanticSearch()
    with open(Path(MOVIES_FILE_PATH), "r") as mfiles:
        docs: list[dict[str, Any]] = json.load(mfiles)["movies"]

    embeddings = sem_search.load_or_create_embeddings(docs)

    print(f"Number of docs: {len(docs)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str) -> None:
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: NDArray[np.float64], vec2: NDArray[np.float64]) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search(query: str, limit: int) -> list[dict[str, float | str]]:
    sem_search = SemanticSearch()
    with open(Path(MOVIES_FILE_PATH), "r") as mfiles:
        docs: list[dict[str, Any]] = json.load(mfiles)["movies"]

    _ = sem_search.load_or_create_embeddings(docs)

    results = sem_search.search(query, limit)
    padding = 4
    desc_limit = 80
    for i, result in enumerate(results):
        left_num = f"{i + 1}."
        print(f"{left_num:<{padding}}{result['title']} (score: {result['score']:.4f})")
        print(f"{' ':<{padding}}{result['description'][:desc_limit]} ...")


def chunk(text: str, chunk_size: int = 200, overlap: int = 0) -> None:
    text_tokens = text.split()
    total_tokens = len(text_tokens)
    i = 0
    curr_idx = 0
    prev_idx = 0
    print(f"Chunking {len(text)} characters")
    while prev_idx < total_tokens:
        i += 1
        curr_idx += chunk_size
        if curr_idx > total_tokens:
            chunk = " ".join(text_tokens[prev_idx:])
        else:
            chunk = " ".join(text_tokens[prev_idx:curr_idx])

        num_prefix = f"{i}."
        print(f"{num_prefix:<3}{chunk}")

        # prev_idx = curr_idx
        prev_idx = curr_idx - overlap


class SemanticSearch:
    def __init__(self) -> None:
        self.embedding_cache_path = Path(CACHE_DIR, "movie_embeddings.npy")

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.embeddings: list[NDArray[np.float64]] | None = None
        self.documents: list[dict[str, Any]] | None = None
        self.document_map: dict[int, dict[str, Any]] | None = {}

    def generate_embedding(self, text: str) -> NDArray[np.float64]:
        if text == "" or text.isspace():
            raise ValueError("Text cannot be empty or whitespace only.")

        embeddings = self.model.encode([text])

        return embeddings[0]

    def build_embeddings(self, documents: list[dict[str, Any]]) -> NDArray[np.float64]:
        self.documents = documents
        doc_text: list[str] = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            doc_text.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(doc_text, show_progress_bar=True)

        if not self.embedding_cache_path.parent.exists():
            self.embedding_cache_path.parent.mkdir()
        with open(self.embedding_cache_path, "wb") as ecache:
            np.save(ecache, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(
        self, documents: list[dict[str, Any]]
    ) -> NDArray[np.float64]:
        if self.embedding_cache_path.exists():
            with open(self.embedding_cache_path, "rb") as ecache:
                self.embeddings = np.load(ecache)
        if self.embeddings is None or (len(self.embeddings) != len(documents)):
            self.embeddings = self.build_embeddings(documents)

        if self.documents is None:
            self.documents = documents
            for doc in self.documents:
                self.document_map[doc["id"]] = doc

        return self.embeddings

    def search(self, query: str, limit: int):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        query_cosine_sim_scores: list[tuple[float, dict[str, int | str]]] = []
        for doc, doc_embedding in zip(self.documents, self.embeddings):
            cosim_score = cosine_similarity(query_embedding, doc_embedding)
            query_cosine_sim_scores.append((cosim_score, doc))

        query_cosine_sim_scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "score": cosim_score,
                "title": doc["title"],
                "description": doc["description"],
            }
            for cosim_score, doc in query_cosine_sim_scores[:limit]
        ]
