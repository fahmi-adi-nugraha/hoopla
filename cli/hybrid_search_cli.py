#!/usr/bin/env python

import sys

from hybrid_search.general import proc
from hybrid_search.opts import get_opts


def main() -> None:
    cli_opts, cli_parser = get_opts()

    try:
        proc(cli_opts, cli_parser)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
