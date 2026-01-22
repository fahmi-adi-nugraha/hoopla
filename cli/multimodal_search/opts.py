from argparse import ArgumentParser, Namespace


def get_opts() -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(description="CLI for performing multimodal search.")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_parser = subparser.add_parser(
        "verify_image_embedding", help="Verify an image embedding"
    )
    verify_image_parser.add_argument(
        "image", type=str, help="Path to the image to embed"
    )

    args = parser.parse_args()

    return args, parser
