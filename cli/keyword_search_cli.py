#!/usr/bin/env python

import argparse
import sys
from pathlib import Path

from keyword_search.inverted_index import InvertedIndex
from keyword_search.keyword_search import (
    get_movies_matching_keywords,
    get_stopwords,
)

MOVIES_FILE_PATH = Path("data/movies.json")

# Hardcode until we know for sure we'll ask user for location
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
    tf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Term whose TF you wish to find")

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
                num_occurrences = inverted_index.get_tf(args.doc_id, args.term)
                doc = inverted_index.docmap[args.doc_id]
                print(
                    f"Number of times '{args.term}' appears in '{doc['title']}': {num_occurrences}"
                )
            except ValueError as e:
                print(f"Value error: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
