from argparse import ArgumentParser, Namespace


def get_opts() -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(description="Keyword Search CLI")
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

    return parser.parse_args(), parser
