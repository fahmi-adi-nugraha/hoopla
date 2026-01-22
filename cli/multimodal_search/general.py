import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from multimodal_search.multimodal_search import MultimodalSearch
from multimodal_search.opts import get_opts

DATA_DIR = "data"


def __get_docs() -> list[dict[str, Any]]:
    with open(Path(DATA_DIR, "movies.json")) as docs:
        return json.load(docs)["movies"]


def verify_image_embedding(
    cli_opts: Namespace, parser: ArgumentParser, mm_searcher: MultimodalSearch
) -> None:
    image_embedding = mm_searcher.embed_image(cli_opts.image)

    print(f"Embedding shape: {image_embedding.shape[0]} dimensions")


def image_search(
    cli_opts: Namespace, parser: ArgumentParser, mm_searcher: MultimodalSearch
) -> None:
    results = mm_searcher.search_with_image(cli_opts.image)

    desc_limit = 100
    padding = 4
    for i, result in enumerate(results):
        num_heading = f"{i + 1}."
        print(
            f"{num_heading:<{padding}}{result['title']} (similarity: {result['cosine_similarity_score']:.3f})"
        )
        print(f"{' ':<{padding}}{result['description'][:desc_limit]}...")


def run() -> None:
    cli_opts, parser = get_opts()

    docs = __get_docs()
    mm_searcher = MultimodalSearch(docs)

    match cli_opts.command:
        case "verify_image_embedding":
            verify_image_embedding(cli_opts, parser, mm_searcher)
        case "image_search":
            image_search(cli_opts, parser, mm_searcher)
        case _:
            parser.print_help()
