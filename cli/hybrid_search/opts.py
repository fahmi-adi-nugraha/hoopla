from argparse import ArgumentParser, Namespace

from hybrid_search.hybrid_search import HYBRID_ALPHA, HYBRID_LIMIT


def get_opts() -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Min-max normalize the BM25 and cosine similarity scores"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", help="Scores to normaalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform a weighted search using the query."
    )
    weighted_search_parser.add_argument(
        "text", type=str, default=HYBRID_ALPHA, help="Search query"
    )
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=HYBRID_ALPHA, help="Alpha value"
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=HYBRID_LIMIT, help="The number of results to show"
    )

    args = parser.parse_args()

    return args, parser
