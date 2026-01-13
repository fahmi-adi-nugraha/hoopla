#!/usr/bin/env python
import os
import sys

from dotenv import load_dotenv
from hybrid_search.general import proc
from hybrid_search.hybrid_search import HybridSearch, load_documents
from hybrid_search.opts import get_opts
from hybrid_search.utils_enhance import QueryEnhancer


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        print(f"Error: Missing Gemini API key")
        sys.exit(1)

    cli_opts, cli_parser = get_opts()

    docs = load_documents()
    search = HybridSearch(docs)
    query_enhancer = QueryEnhancer(api_key)

    try:
        proc(cli_opts, cli_parser, search, query_enhancer)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
