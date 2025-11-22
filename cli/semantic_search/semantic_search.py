import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


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


class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def generate_embedding(self, text: str):
        if text == "" or text.isspace():
            raise ValueError("Text cannot be empty or whitespace only.")

        embeddings = self.model.encode([text])

        return embeddings[0]
