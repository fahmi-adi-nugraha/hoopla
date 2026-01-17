from argparse import ArgumentParser, Namespace


def get_opts() -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()

    return args, parser
