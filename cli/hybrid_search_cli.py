#!/usr/bin/env python

import sys

from hybrid_search.general import proc
from hybrid_search.hybrid_search import HybridSearch, load_documents
from hybrid_search.opts import get_opts


def main() -> None:
    cli_opts, cli_parser = get_opts()

    docs = load_documents()
    search = HybridSearch(docs)

    try:
        proc(cli_opts, cli_parser, search)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
