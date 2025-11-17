import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .inverted_index import BM25_B, BM25_K1, InvertedIndex
from .keyword_search import get_movies_matching_keywords, get_stopwords

MOVIES_FILE_PATH = Path("data/movies.json")
STOPWORDS_FILE = Path("data/stopwords.txt")


def load_index(inverted_index: InvertedIndex) -> None:
    try:
        inverted_index.load()
    except FileNotFoundError as e:
        print(f"Could not find index or docmap: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def search_command(inv_idx: InvertedIndex, query: str) -> None:
    load_index(inv_idx)
    print(f"Searching for: {query}")
    stopwords = get_stopwords(STOPWORDS_FILE)
    movies = get_movies_matching_keywords(inv_idx, query, stopwords)
    for i, movie in enumerate(sorted(movies, key=lambda m: m["id"])):
        print(f"{i + 1}. {movie['title']}")


def build_command(inv_idx: InvertedIndex) -> None:
    stopwords = get_stopwords(STOPWORDS_FILE)
    inv_idx.build(MOVIES_FILE_PATH, stopwords)


def tf_command(inv_idx: InvertedIndex, doc_id: int, term: str) -> None:
    load_index(inv_idx)
    tf = inv_idx.get_tf(doc_id, term)
    print(f"TF score of '{term}' in '{doc_id}': {tf}")


def idf_command(inv_idx: InvertedIndex, term: str) -> None:
    load_index(inv_idx)
    idf = inv_idx.get_idf(term)
    print(f"IDF score of '{term}': {idf}")


def tfidf_command(inv_idx: InvertedIndex, doc_id: int, term: str) -> None:
    load_index(inv_idx)
    tf = inv_idx.get_tf(doc_id, term)
    idf = inv_idx.get_idf(term)
    tfidf = tf * idf
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tfidf:.2f}")


def bm25_tf_command(
    inv_idx: InvertedIndex,
    doc_id: int,
    term: str,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> None:
    load_index(inv_idx)
    bm25tf = inv_idx.get_bm25_tf(doc_id, term, k1, b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")


def bm25_idf_command(inv_idx: InvertedIndex, term: str) -> None:
    load_index(inv_idx)
    bm25idf = inv_idx.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")


def proc(inv_idx: InvertedIndex, args: Namespace, arg_parser: ArgumentParser) -> None:
    match args.command:
        case "search":
            search_command(inv_idx, args.query)
        case "build":
            build_command(inv_idx)
        case "tf":
            tf_command(inv_idx, args.tf_doc_id, args.tf_term)
        case "idf":
            idf_command(inv_idx, args.idf_term)
        case "tfidf":
            tfidf_command(inv_idx, args.tfidf_doc_id, args.tfidf_term)
        case "bm25tf":
            bm25_tf_command(
                inv_idx,
                args.bm25tf_doc_id,
                args.bm25tf_term,
                args.bm25tf_k1,
                args.bm25tf_b,
            )
        case "bm25idf":
            bm25_idf_command(inv_idx, args.bm25idf_term)
        case _:
            arg_parser.print_help()
