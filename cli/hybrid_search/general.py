from argparse import ArgumentParser, Namespace

from hybrid_search.hybrid_search import HybridSearch
from hybrid_search.utils_enhance import QueryEnhancer
from sympy.functions.elementary.tests.test_trigonometric import en


def normalize(searcher: HybridSearch, scores: list[int | float]) -> None:
    scores_normed = searcher.normalize(scores)
    if scores_normed:
        for score in scores_normed:
            print(f"* {score:.4f}")


def weighted_search(searcher: HybridSearch, query: str, alpha: float, limit: int):
    results = searcher.weighted_search(query, alpha, limit)
    padding = 4
    for i, result in enumerate(results):
        left_num = f"{i + 1}."
        print(f"{left_num:<{padding}}{result['title']}")
        print(f"{' ':<{padding}}Hybrid Score: {result['hybrid_score']:.4f}")
        print(
            f"{' ':<{padding}}BM25: {result['bm25_score']:.4f}, Semantic: {result['semantic_score']:.4f}"
        )
        print(f"{' ':<{padding}}{result['description']}...")
    pass


def rrf_search(
    searcher: HybridSearch,
    query_enhancer: QueryEnhancer,
    query: str,
    k: int,
    limit: int,
    enhancement_type: str | None,
):
    query_enhanced = query_enhancer.enhance(query, enhancement_type)
    results = searcher.rrf_search(query, k, limit)
    padding = 4
    if enhancement_type is not None and enhancement_type:
        print(f"Enhanced query ({enhancement_type}): '{query}' -> '{query_enhanced}'")
    for i, result in enumerate(results):
        left_num = f"{i + 1}."
        print(f"{left_num:<{padding}}{result['title']}")
        print(f"{' ':<{padding}}RRF Score: {result['rrf_score']:.4f}")
        print(
            f"{' ':<{padding}}BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
        )
        print(f"{' ':<{padding}}{result['description']}...")


def proc(
    cli_opts: Namespace,
    opt_parser: ArgumentParser,
    searcher: HybridSearch,
    query_enhancer: QueryEnhancer,
) -> None:
    match cli_opts.command:
        case "normalize":
            normalize(searcher, cli_opts.scores)
        case "weighted-search":
            weighted_search(searcher, cli_opts.text, cli_opts.alpha, cli_opts.limit)
        case "rrf-search":
            rrf_search(
                searcher,
                query_enhancer,
                cli_opts.text,
                cli_opts.k,
                cli_opts.limit,
                cli_opts.enhance,
            )
        case _:
            opt_parser.print_help()
