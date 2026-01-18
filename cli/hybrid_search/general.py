from argparse import ArgumentParser, Namespace
from logging import Logger

from hybrid_search.hybrid_search import HybridSearch
from hybrid_search.utils_enhance import QueryEnhancer
from hybrid_search.utils_rerank import LLMReranker

HYBRID_DESCRIPTION_LENGTH = 100


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
        print(f"{' ':<{padding}}{result['description'][:HYBRID_DESCRIPTION_LENGTH]}...")
    pass


def rrf_search(
    logger: Logger,
    searcher: HybridSearch,
    query_enhancer: QueryEnhancer,
    reranker: LLMReranker,
    query: str,
    k: int,
    limit: int,
    query_enhancement_method: str | None,
    reranking_method: str | None,
):
    logger.debug("original query: '%s'", query)

    query_enhanced = query_enhancer.enhance(query, query_enhancement_method)

    logger.debug("enhanced query: '%s'", query_enhanced)

    new_limit = limit * 5
    results = searcher.rrf_search(query_enhanced, k, new_limit)

    logger.debug(
        "results of RRF search: %s", ", ".join([result["title"] for result in results])
    )

    reranking = False
    if reranking_method is not None and reranking_method:
        reranking = True
        results = reranker.rerank(query_enhanced, results, limit, reranking_method)

        logger.debug(
            "results of RRF search after reranking: %s",
            ", ".join([result["title"] for result in results]),
        )

        print(f"Reranking top {limit} results using {reranking_method}...")
    padding = 4
    if query_enhancement_method is not None and query_enhancement_method:
        print(
            f"Enhanced query ({query_enhancement_method}): '{query}' -> '{query_enhanced}'"
        )
    print(f"Reciprocal Rank Fusion results for '{query_enhanced}' (k={k}):\n")
    for i, result in enumerate(results):
        left_num = f"{i + 1}."
        print(f"{left_num:<{padding}}{result['title']}")
        if reranking:
            match reranking_method:
                case "individual":
                    print(f"{' ':<{padding}}Rerank Score: {result['rerank_score']:.4f}")
                case "batch":
                    print(f"{' ':<{padding}}Rerank Rank: {result['rerank_rank']}")
                case "cross_encoder":
                    print(
                        f"{' ':<{padding}}Cross Encoder Score: {result['cross_encoder_score']:.4f}"
                    )
                case _:
                    pass
        print(f"{' ':<{padding}}RRF Score: {result['rrf_score']:.4f}")
        print(
            f"{' ':<{padding}}BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}"
        )
        print(f"{' ':<{padding}}{result['description'][:HYBRID_DESCRIPTION_LENGTH]}...")


def run(
    cli_opts: Namespace,
    opt_parser: ArgumentParser,
    logger: Logger,
    searcher: HybridSearch,
    query_enhancer: QueryEnhancer,
    reranker: LLMReranker,
) -> None:
    match cli_opts.command:
        case "normalize":
            normalize(searcher, cli_opts.scores)
        case "weighted-search":
            weighted_search(searcher, cli_opts.text, cli_opts.alpha, cli_opts.limit)
        case "rrf-search":
            rrf_search(
                logger,
                searcher,
                query_enhancer,
                reranker,
                cli_opts.text,
                cli_opts.k,
                cli_opts.limit,
                cli_opts.enhance,
                cli_opts.rerank_method,
            )
        case _:
            opt_parser.print_help()
