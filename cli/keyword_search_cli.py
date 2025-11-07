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


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser(
        "build", help="Build inverted index for movie data"
    )

    args = parser.parse_args()

    inverted_index = InvertedIndex()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            try:
                inverted_index.load()
            except FileNotFoundError as e:
                print(f"could not find index or docmap: {e}")
                sys.exit(1)

            stopwords = get_stopwords(STOPWORDS_FILE)
            movies = get_movies_matching_keywords(inverted_index, args.query, stopwords)
            for i, movie in enumerate(sorted(movies, key=lambda m: m["id"])):
                print(f"{i + 1}. {movie['title']}")
        case "build":
            stopwords = get_stopwords(STOPWORDS_FILE)
            inverted_index.build(MOVIES_FILE_PATH, stopwords)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
