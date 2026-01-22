from pathlib import Path

from PIL import Image
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "clip-ViT-B-32"


class MultimodalSearch:
    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name or DEFAULT_MODEL
        self.model = SentenceTransformer(self.model_name)

    def embed_image(self, image_path: str):
        with Image.open(Path(image_path)) as im:
            image_embedding = self.model.encode([im])
        return image_embedding[0]
