from argparse import ArgumentParser, Namespace

from hybrid_search.hybrid_search import normalize


def proc(cli_opts: Namespace, opt_parser: ArgumentParser) -> None:
    match cli_opts.command:
        case "normalize":
            scores_normed = normalize(cli_opts.scores)
            if scores_normed:
                for score in scores_normed:
                    print(f"* {score:.4f}")
        case _:
            opt_parser.print_help()
