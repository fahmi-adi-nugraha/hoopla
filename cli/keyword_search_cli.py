#!/usr/bin/env python

import argparse
import math
import sys
from pathlib import Path

from keyword_search.inverted_index import InvertedIndex
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

    args = parser.parse_args()

    inverted_index = InvertedIndex()

    match args.command:
        case "search":
            load_index(inverted_index)
            print(f"Searching for: {args.query}")
            stopwords = get_stopwords(STOPWORDS_FILE)
            movies = get_movies_matching_keywords(inverted_index, args.query, stopwords)
            for i, movie in enumerate(sorted(movies, key=lambda m: m["id"])):
                print(f"{i + 1}. {movie['title']}")
        case "build":
            stopwords = get_stopwords(STOPWORDS_FILE)
            inverted_index.build(MOVIES_FILE_PATH, stopwords)
        case "tf":
            load_index(inverted_index)
            try:
                num_occurrences = inverted_index.get_tf(args.tf_doc_id, args.tf_term)
                doc = inverted_index.docmap[args.tf_doc_id]
                print(
                    f"Number of times '{args.tf_term}' appears in '{doc['title']}': {num_occurrences}"
                )
            except ValueError as e:
                print(f"Value error: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)
        case "idf":
            load_index(inverted_index)
            if len(args.idf_term.split()) > 1:
                print("Value error: 'idf' only accepts one term")
                sys.exit(1)
            doc_count = len(inverted_index.docmap)
            term_doc_count = len(inverted_index.get_documents(args.idf_term))
            idf = math.log((doc_count + 1) / (term_doc_count + 1))
            print(f"IDF of '{args.idf_term}': {idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
