from argparse import ArgumentParser, Namespace

from multimodal_search.multimodal_search import MultimodalSearch
from multimodal_search.opts import get_opts


def verify_image_embedding(
    cli_opts: Namespace, parser: ArgumentParser, mm_searcher: MultimodalSearch
) -> None:
    image_embedding = mm_searcher.embed_image(cli_opts.image)

    print(f"Embedding shape: {image_embedding.shape[0]} dimensions")


def run() -> None:
    cli_opts, parser = get_opts()

    mm_searcher = MultimodalSearch()

    match cli_opts.command:
        case "verify_image_embedding":
            verify_image_embedding(cli_opts, parser, mm_searcher)
        case _:
            parser.print_help()
