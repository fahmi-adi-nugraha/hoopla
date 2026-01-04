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

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Create embedding for the query"
    )
    embed_query_parser.add_argument("text", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Perform semantic search")
    search_parser.add_argument("text", type=str, help="Query to embed")
    search_parser.add_argument(
        "--limit", "-n", type=int, default=5, help="Number of results to display"
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Create smaller chunks from the provided text"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", "-n", type=int, default=200, help="Size of each chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Amount of overlap between each chunk"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Create smaller chunks from the provided text"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        "-n",
        type=int,
        default=200,
        help="Maximum size of each chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Amount of overlap between each chunk",
    )

    embed_chunk_parser = subparsers.add_parser(
        "embed_chunks", help="Chunk documents and create embeddings for each chunk"
    )

    args = parser.parse_args()

    return args, parser
