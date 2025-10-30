#!/usr/bin/env python

import argparse
import json
import string
from functools import reduce
from pathlib import Path
from typing import Any, Callable

MOVIES_FILE_PATH = Path("data/movies.json")


# Consider moving all the text cleaning functions to their own module or even package


def remove_punctuation(text: str) -> str:
    str_trans_map = str.maketrans("", "", string.punctuation)
    return text.translate(str_trans_map)


def convert_to_lower(text: str) -> str:
    return text.lower()


def clean_text(text: str) -> str:
    func_list: list[Callable[[str], str]] = [
        convert_to_lower,
        remove_punctuation,
    ]
    return reduce(lambda acc, func: func(acc), func_list, text)


def get_movies_matching_keywords(movie_data_path: Path, keywords: str) -> list[str]:
    with open(movie_data_path, "r") as movie_data_file:
        movie_data: dict[str, list[dict[str, Any]]] = json.load(movie_data_file)
    movie_matches: list[dict[str, Any]] = []
    keywords_cleaned = clean_text(keywords)
    for movie in movie_data["movies"]:
        movie_title_cleaned = clean_text(movie["title"])
        if keywords_cleaned in movie_title_cleaned:
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
            movies = get_movies_matching_keywords(MOVIES_FILE_PATH, args.query)
            for i, movie in enumerate(sorted(movies, key=lambda m: m["id"])[:5]):
                print(f"{i + 1}. {movie['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
