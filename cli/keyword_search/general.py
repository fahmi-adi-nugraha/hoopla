import sys
from argparse import ArgumentParser, Namespace

from .inverted_index import BM25_B, BM25_K1, InvertedIndex
from .text_processing.text_processing import TextClean

BM25_SEARCH_RESULTS_LIMIT = 5


def _get_movies_matching_keywords(
    movie_idx: InvertedIndex, text_cleaner: TextClean, keywords: str
) -> list[dict[str, int | str]]:
    text_cleaner.result = keywords
    keywords_clean = (
        text_cleaner.convert_to_lower()
        .remove_punctuation()
        .tokenize()
        .remove_stop_words()
        .stem_tokens()
        .result
    )
    movie_matches: list[dict[str, int | str]] = []
    for keyword in keywords_clean:
        doc_indexes = movie_idx.get_documents(keyword)
        for doc_index in doc_indexes:
            movie_matches.append(movie_idx.docmap[doc_index])
            if len(movie_matches) == 5:
                return movie_matches
    return movie_matches


def load_stopwords(self) -> None:
    with open(self.stopwords_file_path, "r") as swf:
        self.stopwords = swf.read().splitlines()


def load_index(inverted_index: InvertedIndex) -> None:
    try:
        inverted_index.load()
    except FileNotFoundError as e:
        print(f"Could not find index or docmap: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def search_command(inv_idx: InvertedIndex, text_cleaner: TextClean, query: str) -> None:
    load_index(inv_idx)
    print(f"Searching for: {query}")
    movies = _get_movies_matching_keywords(inv_idx, text_cleaner, query)
    for i, movie in enumerate(sorted(movies, key=lambda m: m["id"])):
        print(f"{i + 1}. {movie['title']}")


def build_command(inv_idx: InvertedIndex) -> None:
    inv_idx.build()


def tf_command(inv_idx: InvertedIndex, doc_id: int, term: str) -> None:
    load_index(inv_idx)
    tf = inv_idx.get_tf(doc_id, term)
    print(f"TF score of '{term}' in '{doc_id}': {tf}")


def idf_command(inv_idx: InvertedIndex, term: str) -> None:
    load_index(inv_idx)
    idf = inv_idx.get_idf(term)
    print(f"IDF score of '{term}': {idf}")


def tfidf_command(inv_idx: InvertedIndex, doc_id: int, term: str) -> None:
    load_index(inv_idx)
    tf = inv_idx.get_tf(doc_id, term)
    idf = inv_idx.get_idf(term)
    tfidf = tf * idf
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tfidf:.2f}")


def bm25_tf_command(
    inv_idx: InvertedIndex,
    doc_id: int,
    term: str,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> None:
    load_index(inv_idx)
    bm25tf = inv_idx.get_bm25_tf(doc_id, term, k1, b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")


def bm25_idf_command(inv_idx: InvertedIndex, term: str) -> None:
    load_index(inv_idx)
    bm25idf = inv_idx.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")


def bm25_search_command(
    inv_idx: InvertedIndex,
    query: str,
    limit: int = BM25_SEARCH_RESULTS_LIMIT,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> None:
    load_index(inv_idx)
    top_matches = inv_idx.bm25_search(query, limit, k1, b)
    for i, (doc_id, bm25_score) in enumerate(top_matches):
        print(
            f"{i + 1}. ({doc_id}) {inv_idx.docmap[doc_id]['title']} - Score: {bm25_score:.2f}"
        )


# def proc(inv_idx: InvertedIndex, args: Namespace, arg_parser: ArgumentParser) -> None:
def proc(
    inv_idx: InvertedIndex,
    text_cleaner: TextClean,
    args: Namespace,
    arg_parser: ArgumentParser,
) -> None:
    match args.command:
        case "search":
            search_command(inv_idx, text_cleaner, args.query)
        case "build":
            build_command(inv_idx)
        case "tf":
            tf_command(inv_idx, args.tf_doc_id, args.tf_term)
        case "idf":
            idf_command(inv_idx, args.idf_term)
        case "tfidf":
            tfidf_command(inv_idx, args.tfidf_doc_id, args.tfidf_term)
        case "bm25tf":
            bm25_tf_command(
                inv_idx,
                args.bm25tf_doc_id,
                args.bm25tf_term,
                args.bm25tf_k1,
                args.bm25tf_b,
            )
        case "bm25idf":
            bm25_idf_command(inv_idx, args.bm25idf_term)
        case "bm25search":
            bm25_search_command(
                inv_idx,
                args.bm25search_query,
                args.limit,
                args.bm25search_k1,
                args.bm25search_b,
            )
        case _:
            arg_parser.print_help()
