from argparse import ArgumentParser, Namespace


def get_opts() -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify the embedding model")

    embed_parser = subparsers.add_parser("embed_text", help="Generate text embedding")
    embed_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings",
        help="Verify that the document embedding process was successful",
    )

    args = parser.parse_args()

    return args, parser
