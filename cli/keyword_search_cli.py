#!/usr/bin/env python

import argparse
import math
import sys
from pathlib import Path

from keyword_search.inverted_index import InvertedIndex, calc_idf, calc_tf_for_doc
from keyword_search.keyword_search import (
    get_movies_matching_keywords,
    get_stopwords,
)

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


def main() -> None:
    # Put this whole thing into a function
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser(
        "build", help="Build inverted index for movie data"
    )

    tf_parser = subparsers.add_parser(
        "tf", help="Get the number of times a term appears in the specfied document"
    )
    tf_parser.add_argument("tf_doc_id", type=int, help="Document id")
    tf_parser.add_argument("tf_term", type=str, help="Term whose TF you wish to find")

    idf_parser = subparsers.add_parser(
        "idf", help="Calculate the IDF of the given term for the specified document"
    )
    idf_parser.add_argument(
        "idf_term", type=str, help="Term whose IDF you wish to find"
    )

    tf_idf_parser = subparsers.add_parser(
        "tfidf",
        help="Calculate the TF-IDF of the given term for the specified document",
    )
    tf_idf_parser.add_argument("tfidf_doc_id", type=int, help="Document id")
    tf_idf_parser.add_argument(
        "tfidf_term", type=str, help="Term whose TF you wish to find"
    )

    args = parser.parse_args()

    inverted_index = InvertedIndex()

    try:
        # Put this whole thing into a function
        match args.command:
            case "search":
                load_index(inverted_index)
                print(f"Searching for: {args.query}")
                stopwords = get_stopwords(STOPWORDS_FILE)
                movies = get_movies_matching_keywords(
                    inverted_index, args.query, stopwords
                )
                for i, movie in enumerate(sorted(movies, key=lambda m: m["id"])):
                    print(f"{i + 1}. {movie['title']}")
            case "build":
                stopwords = get_stopwords(STOPWORDS_FILE)
                inverted_index.build(MOVIES_FILE_PATH, stopwords)
            case "tf":
                load_index(inverted_index)
                tf, doc_title = calc_tf_for_doc(
                    inverted_index, args.tf_doc_id, args.tf_term
                )
                print(
                    f"Number of times '{args.tf_term}' appears in '{doc_title}': {tf}"
                )
            case "idf":
                load_index(inverted_index)
                idf = calc_idf(inverted_index, args.idf_term)
                print(f"IDF of '{args.idf_term}': {idf:.2f}")
            case "tfidf":
                load_index(inverted_index)
                tf, doc_title = calc_tf_for_doc(
                    inverted_index, args.tfidf_doc_id, args.tfidf_term
                )
                idf = calc_idf(inverted_index, args.tfidf_term)
                tfidf = tf * idf
                print(
                    f"TF-IDF score of '{args.tfidf_term}' in document '{doc_title}': {tfidf:.2f}"
                )
            case _:
                parser.print_help()
    except ValueError as e:
        print(f"Value error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
