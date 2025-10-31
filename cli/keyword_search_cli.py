#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import Any

from text_processing.text_processing import clean_text

MOVIES_FILE_PATH = Path("data/movies.json")

# Hardcode until we know for sure we'll ask user for location
STOPWORDS_FILE = Path("data/stopwords.txt")


def match_movie_title(keyword_tokens: list[str], movie_title_tokens: list[str]) -> bool:
    for movie_title_tok in movie_title_tokens:
        for keyword_tok in keyword_tokens:
            if keyword_tok in movie_title_tok:
                return True
    return False


def get_stopwords(stop_words_file: Path) -> list[str]:
    stop_words: list[str]
    with open(stop_words_file, "r") as swf:
        stop_words = swf.read().splitlines()
    return stop_words


def get_movies_matching_keywords(
    movie_data_path: Path, keywords: str, stop_words: list[str]
) -> list[str]:
    with open(movie_data_path, "r") as movie_data_file:
        movie_data: dict[str, list[dict[str, Any]]] = json.load(movie_data_file)
    movie_matches: list[dict[str, Any]] = []
    keywords_cleaned = clean_text(keywords, stop_words)
    for movie in movie_data["movies"]:
        movie_title_cleaned = clean_text(movie["title"], stop_words)
        if match_movie_title(keywords_cleaned, movie_title_cleaned):
            movie_matches.append(movie)
    return movie_matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            stop_words = get_stopwords(STOPWORDS_FILE)
            movies = get_movies_matching_keywords(
                MOVIES_FILE_PATH, args.query, stop_words
            )
            for i, movie in enumerate(sorted(movies, key=lambda m: m["id"])[:5]):
                print(f"{i + 1}. {movie['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
