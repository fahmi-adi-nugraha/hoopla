import json
from pathlib import Path
from typing import Any

from .text_processing.text_processing import clean_text


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
) -> list[dict[str, Any]]:
    with open(movie_data_path, "r") as movie_data_file:
        movie_data: dict[str, list[dict[str, Any]]] = json.load(movie_data_file)
    movie_matches: list[dict[str, Any]] = []
    keywords_cleaned = clean_text(keywords, stop_words)
    for movie in movie_data["movies"]:
        movie_title_cleaned = clean_text(movie["title"], stop_words)
        if match_movie_title(keywords_cleaned, movie_title_cleaned):
            movie_matches.append(movie)
    return movie_matches


def get_movies(
    movie_data_path: Path, stop_words_file: Path, keywords: str
) -> list[dict[str, Any]]:
    stopwords = get_stopwords(stop_words_file)
    return get_movies_matching_keywords(movie_data_path, keywords, stopwords)
