from argparse import ArgumentParser, Namespace

from hybrid_search.hybrid_search import HybridSearch


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


def proc(
    cli_opts: Namespace, opt_parser: ArgumentParser, searcher: HybridSearch
) -> None:
    match cli_opts.command:
        case "normalize":
            normalize(searcher, cli_opts.scores)
        case "weighted-search":
            weighted_search(searcher, cli_opts.text, cli_opts.alpha, cli_opts.limit)
        case _:
            opt_parser.print_help()
