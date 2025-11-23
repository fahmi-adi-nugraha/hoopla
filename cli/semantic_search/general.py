from argparse import ArgumentParser, Namespace

from semantic_search.semantic_search import (
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def proc(cli_opts: Namespace, opt_parser: ArgumentParser):
    match cli_opts.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(cli_opts.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(cli_opts.text)
        case _:
            opt_parser.print_help()
