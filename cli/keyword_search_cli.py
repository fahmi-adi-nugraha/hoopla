#!/usr/bin/env python

import sys

from keyword_search.general import proc
from keyword_search.inverted_index import InvertedIndex
from keyword_search.opts import get_opts
from keyword_search.text_processing.text_processing import TextProcessingContext
from nltk.stem import PorterStemmer


def main() -> None:
    # Might even wrap this in a `run` function
    args, parser = get_opts()
    stemmer = PorterStemmer()
    text_processing_context = TextProcessingContext(stemmer)
    inverted_index = InvertedIndex(stemmer, text_processing_context)

    try:
        proc(inverted_index, text_processing_context, args, parser)
    except ValueError as e:
        print(f"Value error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
