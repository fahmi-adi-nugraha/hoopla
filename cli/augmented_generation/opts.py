from argparse import ArgumentParser, Namespace


def get_opts() -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    rag_summarize_parser = subparsers.add_parser(
        "summarize",
        help="Perform RAG (search + generate answer) to generate a comprehensive summary",
    )
    rag_summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_summarize_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="The number of results to use from the document search",
    )

    args = parser.parse_args()

    return args, parser
