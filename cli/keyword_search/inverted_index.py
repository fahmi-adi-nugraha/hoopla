import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from nltk.stem import PorterStemmer

from .text_processing.text_processing import (
    clean_text,
    clean_text_finish,
    clean_text_up_to_tokenize,
    tokenize,
)

CACHE_DIR = "./cache"
BM25_K1 = 1.5
BM25_B = 0.75


# TODO: Figure out how to make the stemming faster for getting the tf and the idf


class InvertedIndex:
    def __init__(self):
        self.index_path: Path = Path(CACHE_DIR, "index.pkl")
        self.docmap_path: Path = Path(CACHE_DIR, "docmap.pkl")
        self.term_freq_path: Path = Path(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path: Path = Path(CACHE_DIR, "doc_lengths.pkl")

        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict[str, int | str]] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.__stemmer = PorterStemmer()
        self.doc_lengths: dict[int, int] = {}

    def __validate_term(self, term: str) -> None:
        term_tok = tokenize(term)
        if len(term_tok) > 1:
            raise ValueError("Expected only one term")

    def __add_document(self, doc_id: int, text: str, stopwords: list[str]) -> None:
        tokens = clean_text_up_to_tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        tokens = clean_text_finish(tokens, stopwords)
        self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        self.__validate_term(term)
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

        self.save(Path(CACHE_DIR))

    def get_tf(self, doc_id: int, term: str) -> int:
        self.__validate_term(term)
        stemmed_term = self.__stemmer.stem(term)
        return self.term_frequencies[doc_id][stemmed_term]

    def get_idf(self, term: str) -> float:
        doc_count = len(self.docmap)
        term_doc_count = len(self.get_documents(term))
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        self.__validate_term(term)
        doc_count = len(self.docmap)
        term_doc_count = len(self.get_documents(term))
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def __get_avg_doc_length(self) -> float:
        num_docs = len(self.doc_lengths)
        if num_docs == 0:
            return 0.0
        doc_lengths_total = 0
        for doc_length in self.doc_lengths.values():
            doc_lengths_total += doc_length
        return doc_lengths_total / num_docs

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        doc_len = self.doc_lengths[doc_id]
        avg_doc_len = self.__get_avg_doc_length()
        length_norm = (1 - b) + (b * (doc_len / avg_doc_len))
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def __serialize(self, file_path: Path, data: dict[Any, Any]) -> None:
        with open(file_path, "wb") as out_file:
            pickle.dump(data, out_file)

    def save(self, cache_dir: Path) -> None:
        if not cache_dir.exists():
            cache_dir.mkdir(exist_ok=True)

        self.__serialize(self.index_path, self.index)
        self.__serialize(self.docmap_path, self.docmap)
        self.__serialize(self.term_freq_path, self.term_frequencies)
        self.__serialize(self.doc_lengths_path, self.doc_lengths)

    def __unserialize(self, file_path: Path) -> dict[Any, Any]:
        if not file_path.exists():
            raise FileNotFoundError(f"could not find file '{file_path}'")
        with open(file_path, "rb") as fp:
            return pickle.load(fp)

    def load(self) -> None:
        self.index = self.__unserialize(self.index_path)
        self.docmap = self.__unserialize(self.docmap_path)
        self.term_frequencies = self.__unserialize(self.term_freq_path)
        self.doc_lengths = self.__unserialize(self.doc_lengths_path)
