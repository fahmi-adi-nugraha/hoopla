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
