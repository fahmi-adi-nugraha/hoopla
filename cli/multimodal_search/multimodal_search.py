from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "clip-ViT-B-32"


class MultimodalSearch:
    def __init__(self, documents: list[dict[str, Any]], model_name: str = "") -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.documents = documents

        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str):
        with Image.open(Path(image_path)) as im:
            image_embedding = self.model.encode([im])
        return image_embedding[0]

    def __cosine_similarity(
        self, vec1: NDArray[np.float64], vec2: NDArray[np.float64]
    ) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def search_with_image(self, image_path: str) -> list[dict[str, Any]]:
        image_embedding = self.embed_image(image_path)
        results: list[dict[str, Any]] = []
        for doc, text_embbedding in zip(self.documents, self.text_embeddings):
            cosim_score = self.__cosine_similarity(image_embedding, text_embbedding)
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "cosine_similarity_score": cosim_score,
                }
            )
        return sorted(
            results, key=lambda x: x["cosine_similarity_score"], reverse=True
        )[:5]
