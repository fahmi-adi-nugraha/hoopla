from argparse import ArgumentParser, Namespace


def get_opts() -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Min-max normalize the BM25 and cosine similarity scores"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", help="Scores to normaalize"
    )

    args = parser.parse_args()

    return args, parser
