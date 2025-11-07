import json
import pickle
from pathlib import Path
from typing import Any

from .text_processing.text_processing import clean_text

DEFAULT_CACHE_DIR = "./cache"


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict[str, int | str]] = {}

    def __add_document(self, doc_id: int, text: str, stopwords: list[str]) -> None:
        tokens = clean_text(text, stopwords)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower(), None)
        if doc_ids is None:
            return []
        return sorted(doc_ids)

    def build(self, movie_data_path: Path, stopwords: list[str]) -> None:
        with open(movie_data_path, "r") as movie_data_file:
            movie_data: dict[str, list[dict[str, Any]]] = json.load(movie_data_file)

        for movie in movie_data["movies"]:
            doc_id: int = movie["id"]
            self.docmap[doc_id] = movie

            self.__add_document(
                doc_id, f"{movie['title']} {movie['description']}", stopwords
            )

        self.save(Path(DEFAULT_CACHE_DIR))

    def save(self, cache_dir: Path) -> None:
        if not cache_dir.exists():
            cache_dir.mkdir(exist_ok=True)

        with open(cache_dir.joinpath("index.pkl"), "wb") as index_file:
            pickle.dump(self.index, index_file)

        with open(cache_dir.joinpath("docmap.pkl"), "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)

    def load(self) -> None:
        cache_dir = Path(DEFAULT_CACHE_DIR)

        idx_file = cache_dir.joinpath("index.pkl")
        if not idx_file.exists():
            raise FileNotFoundError(f"could not find file '{idx_file}'")
        with open(idx_file, "rb") as idx:
            self.index = pickle.load(idx)

        docmap_file = cache_dir.joinpath("docmap.pkl")
        if not docmap_file.exists():
            raise FileNotFoundError(f"could not find file '{docmap_file}'")
        with open(cache_dir.joinpath("docmap.pkl"), "rb") as dcm:
            self.docmap = pickle.load(dcm)
