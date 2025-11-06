#!/usr/bin/env python

import argparse
from pathlib import Path

from keyword_search.keyword_search import get_movies

MOVIES_FILE_PATH = Path("data/movies.json")

# Hardcode until we know for sure we'll ask user for location
STOPWORDS_FILE = Path("data/stopwords.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies = get_movies(MOVIES_FILE_PATH, STOPWORDS_FILE, args.query)
            for i, movie in enumerate(sorted(movies, key=lambda m: m["id"])[:5]):
                print(f"{i + 1}. {movie['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
