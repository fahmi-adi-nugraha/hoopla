from pathlib import Path

from .inverted_index import InvertedIndex
from .text_processing.text_processing import clean_text


def get_stopwords(stop_words_file: Path) -> list[str]:
    stop_words: list[str]
    with open(stop_words_file, "r") as swf:
        stop_words = swf.read().splitlines()
    return stop_words


def get_movies_matching_keywords(
    movie_idx: InvertedIndex, keywords: str, stop_words: list[str]
) -> list[dict[str, int | str]]:
    keywords_clean = clean_text(keywords, stop_words)
    movie_matches: list[dict[str, int | str]] = []
    for keyword in keywords_clean:
        doc_indexes = movie_idx.get_documents(keyword)
        for doc_index in doc_indexes:
            movie_matches.append(movie_idx.docmap[doc_index])
            if len(movie_matches) == 5:
                return movie_matches
    return movie_matches
