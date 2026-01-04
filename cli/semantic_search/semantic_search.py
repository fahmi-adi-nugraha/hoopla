import json
import re
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
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than the chunk-size")

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

        prev_idx = curr_idx - overlap


def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be smaller than the max-chunk-size")

    pre_chunked_text = re.split(r"(?<=[.!?])\s+", text)
    total_pre_chunks = len(pre_chunked_text)
    i = 0
    curr_idx = 0
    prev_idx = 0
    chunks: list[str] = []
    while curr_idx < total_pre_chunks:
        i += 1

        if i == 1:
            curr_idx += max_chunk_size
        else:
            curr_idx += max_chunk_size - overlap

        if curr_idx > total_pre_chunks:
            chunk = " ".join(pre_chunked_text[prev_idx:])
        else:
            chunk = " ".join(pre_chunked_text[prev_idx:curr_idx])

        chunks.append(chunk)

        if max_chunk_size == total_pre_chunks:
            break

        prev_idx = curr_idx - overlap

    return chunks


def semantic_chunk_pretty(text: str, max_chunk_size: int = 4, overlap: int = 0) -> None:
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be smaller than the max-chunk-size")

    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        num_prefix = f"{i}."
        print(f"{num_prefix:<3}{chunk}")


def embed_chunks() -> None:
    chunked_sem_search = ChunkedSemanticSearch()
    with open(Path(MOVIES_FILE_PATH), "r") as mfiles:
        docs: list[dict[str, Any]] = json.load(mfiles)["movies"]

    embeddings = chunked_sem_search.load_or_create_chunk_embeddings(docs)

    print(f"Generated {len(embeddings)} chunked embeddings")


class SemanticSearch:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.embedding_cache_path = Path(CACHE_DIR, "movie_embeddings.npy")

        self.model = SentenceTransformer(model_name)
        self.embeddings: list[NDArray[np.float64]] | None = None
        self.documents: list[dict[str, Any]] | None = None
        self.document_map: dict[int, dict[str, Any]] | None = None

    def generate_embedding(self, text: str) -> NDArray[np.float64]:
        if text == "" or text.isspace():
            raise ValueError("Text cannot be empty or whitespace only.")

        embeddings = self.model.encode([text])

        return embeddings[0]

    def build_embeddings(self, documents: list[dict[str, Any]]) -> NDArray[np.float64]:
        self.documents = documents
        doc_text: list[str] = []
        self.document_map = {}
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
                self.embeddings = np.load(ecache, allow_pickle=True)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        super().__init__(model_name)
        self.chunk_embeddings_cache_path = Path(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_cache_path = Path(CACHE_DIR, "chunk_metadata.json")

        self.chunk_embeddings = None
        # self.chunk_metadata: list[dict[str, int]] = None
        self.chunk_metadata: list[dict[str, int]] = []

    def build_chunk_embeddings(
        self, documents: list[dict[str, Any]]
    ) -> list[NDArray[np.float64]]:
        self.documents = documents
        doc_chunks: list[str] = []
        self.document_map = {}
        # doc_metadata: dict[str, int] = {}
        for doc in self.documents:
            description = doc.get("description")
            if description is None:
                continue
            description_chunks = semantic_chunk(description, 4, 1)
            for i, desc_chunk in enumerate(description_chunks):
                doc_chunks.append(desc_chunk)
                self.chunk_metadata.append(
                    {
                        "movie_idx": doc["id"],
                        "chunk_idx": i + 1,
                        "total_chunks": len(description_chunks),
                    }
                )

            self.document_map[doc["id"]] = doc

        # self.chunk_metadata = doc_metadata
        self.chunk_embeddings = self.model.encode(doc_chunks, show_progress_bar=True)

        if not self.chunk_embeddings_cache_path.parent.exists():
            self.chunk_embeddings_cache_path.parent.mkdir()

        # print(f"Saving embeddings to {self.chunk_embeddings_cache_path}...")
        with open(self.chunk_embeddings_cache_path, "wb") as ecache:
            np.save(ecache, self.chunk_embeddings)

        # print(f"Saving metadata to {self.chunk_metadata_cache_path}...")
        with open(self.chunk_metadata_cache_path, "w") as mcache:
            json.dump(
                {"chunks": self.chunk_metadata, "total_chunks": len(doc_chunks)},
                mcache,
                indent=2,
            )

        # print("Saved all essential data. Returning embeddings.")

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        if self.chunk_embeddings_cache_path.exists():
            with open(self.chunk_embeddings_cache_path, "rb") as ecache:
                self.chunk_embeddings = np.load(ecache, allow_pickle=True)

        # Need to figure out another way to check whether the chunked embeddings for the
        # current docs are correct. For now, will just check if they're None.
        # if self.chunk_embeddings is None or (
        #     len(self.chunk_embeddings) != len(documents)
        # ):
        if self.chunk_embeddings is None:
            self.chunk_embeddings = self.build_chunk_embeddings(documents)

        if self.documents is None:
            self.documents = documents
            self.document_map = {}
            for doc in self.documents:
                self.document_map[doc["id"]] = doc

        return self.chunk_embeddings
