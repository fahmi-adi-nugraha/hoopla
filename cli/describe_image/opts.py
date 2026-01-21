from argparse import ArgumentParser, Namespace


def get_opts() -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(description="Use an LLM to describe an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query to rewrite based on the image",
    )

    args = parser.parse_args()

    return args, parser
