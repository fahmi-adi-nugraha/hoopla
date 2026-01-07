from argparse import ArgumentParser, Namespace


def proc(cli_opts: Namespace, opt_parser: ArgumentParser) -> None:
    match cli_opts.command:
        case _:
            opt_parser.print_help()
