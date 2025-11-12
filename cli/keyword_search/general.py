import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .inverted_index import InvertedIndex
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


def proc(inv_idx: InvertedIndex, args: Namespace, arg_parser: ArgumentParser) -> None:
    match args.command:
        case "search":
            load_index(inv_idx)
            print(f"Searching for: {args.query}")
            stopwords = get_stopwords(STOPWORDS_FILE)
            movies = get_movies_matching_keywords(inv_idx, args.query, stopwords)
            for i, movie in enumerate(sorted(movies, key=lambda m: m["id"])):
                print(f"{i + 1}. {movie['title']}")
        case "build":
            stopwords = get_stopwords(STOPWORDS_FILE)
            inv_idx.build(MOVIES_FILE_PATH, stopwords)
        case "tf":
            load_index(inv_idx)
            tf = inv_idx.get_tf(args.tf_doc_id, args.tf_term)
            doc_title = inv_idx.docmap[args.tf_doc_id]["title"]
            print(f"Number of times '{args.tf_term}' appears in '{doc_title}': {tf}")
        case "idf":
            load_index(inv_idx)
            idf = inv_idx.get_idf(args.idf_term)
            print(f"IDF score of '{args.idf_term}': {idf:.2f}")
        case "tfidf":
            load_index(inv_idx)
            tf = inv_idx.get_tf(args.tfidf_doc_id, args.tfidf_term)
            doc_title = inv_idx.docmap[args.tfidf_doc_id]["title"]
            idf = inv_idx.get_idf(args.tfidf_term)
            tfidf = tf * idf
            print(
                f"TF-IDF score of '{args.tfidf_term}' in document '{doc_title}': {tfidf:.2f}"
            )
        case "bm25tf":
            load_index(inv_idx)
            bm25tf = inv_idx.get_bm25_tf(args.bm25tf_doc_id, args.bm25tf_term)
            print(
                f"BM25 TF score of '{args.bm25tf_term}' in document '{args.bm25tf_doc_id}': {bm25tf:.2f}"
            )
        case "bm25idf":
            load_index(inv_idx)
            bm25idf = inv_idx.get_bm25_idf(args.bm25idf_term)
            print(f"BM25 IDF score of '{args.bm25idf_term}': {bm25idf:.2f}")
        case _:
            arg_parser.print_help()
