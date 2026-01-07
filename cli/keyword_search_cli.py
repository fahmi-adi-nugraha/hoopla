#!/usr/bin/env python

import sys

from keyword_search.general import proc
from keyword_search.inverted_index import InvertedIndex
from keyword_search.opts import get_opts
from keyword_search.text_processing.text_processing import TextClean


def main() -> None:
    # Might even wrap this in a `run` function
    args, parser = get_opts()
    inverted_index = InvertedIndex()
    text_cleaner = TextClean()

    try:
        # proc(inverted_index, args, parser)
        proc(inverted_index, text_cleaner, args, parser)
    except ValueError as e:
        print(f"Value error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
