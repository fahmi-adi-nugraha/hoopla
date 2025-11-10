import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from nltk.stem import PorterStemmer

from .text_processing.text_processing import clean_text, tokenize

DEFAULT_CACHE_DIR = "./cache"


# TODO: Figure out how to make the stemming faster for getting the tf and the idf


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict[str, int | str]] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.__stemmer = PorterStemmer()

    def __add_document(self, doc_id: int, text: str, stopwords: list[str]) -> None:
        tokens = clean_text(text, stopwords)
        self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        term_tok = tokenize(term)
        if len(term_tok) > 1:
            raise ValueError("Expected only one term")
        stemmed_term = self.__stemmer.stem(term)
        doc_ids = self.index.get(stemmed_term, None)
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

    def get_tf(self, doc_id: int, term: str) -> int:
        term_tok = tokenize(term)
        if len(term_tok) > 1:
            raise ValueError("Expected only one term")
        stemmed_term = self.__stemmer.stem(term)
        return self.term_frequencies[doc_id][stemmed_term]

    def __serialize(self, file_path: Path, data: dict[Any, Any]) -> None:
        with open(file_path, "wb") as out_file:
            pickle.dump(data, out_file)

    def save(self, cache_dir: Path) -> None:
        if not cache_dir.exists():
            cache_dir.mkdir(exist_ok=True)

        self.__serialize(cache_dir.joinpath("index.pkl"), self.index)
        self.__serialize(cache_dir.joinpath("docmap.pkl"), self.docmap)
        self.__serialize(
            cache_dir.joinpath("term_frequencies.pkl"), self.term_frequencies
        )

    def __unserialize(self, file_path: Path) -> dict[Any, Any]:
        if not file_path.exists():
            raise FileNotFoundError(f"could not find file '{file_path}'")
        with open(file_path, "rb") as fp:
            return pickle.load(fp)

    def load(self) -> None:
        cache_dir = Path(DEFAULT_CACHE_DIR)

        idx_file = cache_dir.joinpath("index.pkl")
        self.index = self.__unserialize(idx_file)

        docmap_file = cache_dir.joinpath("docmap.pkl")
        self.docmap = self.__unserialize(docmap_file)

        term_freq_file = cache_dir.joinpath("term_frequencies.pkl")
        self.term_frequencies = self.__unserialize(term_freq_file)


def calc_tf_for_doc(invidx: InvertedIndex, doc_id: int, term: str) -> tuple[int, str]:
    num_occurrences = invidx.get_tf(doc_id, term)
    doc = invidx.docmap[doc_id]
    return num_occurrences, doc["title"]


def calc_idf(invidx: InvertedIndex, term: str) -> float:
    doc_count = len(invidx.docmap)
    term_doc_count = len(invidx.get_documents(term))
    idf = math.log((doc_count + 1) / (term_doc_count + 1))
    return idf
