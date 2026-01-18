#!/usr/bin/env python
import os
import sys

from dotenv import load_dotenv
from hybrid_search.general import run
from hybrid_search.hybrid_search import HybridSearch
from hybrid_search.opts import get_opts
from hybrid_search.utils_enhance import QueryEnhancer
from hybrid_search.utils_logging import new_logger
from hybrid_search.utils_rerank import LLMReranker


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key is None:
        print("Error: Missing Gemini API key")
        sys.exit(1)

    cli_opts, cli_parser = get_opts()

    logger = new_logger()

    search = HybridSearch()
    query_enhancer = QueryEnhancer(api_key)
    reranker = LLMReranker(api_key)

    try:
        run(cli_opts, cli_parser, logger, search, query_enhancer, reranker)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
