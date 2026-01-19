from augmented_generation.augmented_generation import RAG
from augmented_generation.opts import get_opts
from hybrid_search.hybrid_search import HybridSearch


def rag(searcher: HybridSearch, rag_client: RAG, query: str) -> None:
    results = searcher.rrf_search(query)
    rag_response = rag_client.answer(query, results)

    print("Search Results:")
    for result in results:
        print(f"\t- {result['title']}")

    print(f"\nRAG Response:\n{rag_response}")


def run(api_key: str) -> None:
    cli_opts, parser = get_opts()

    searcher = HybridSearch()
    rag_client = RAG(api_key=api_key)

    match cli_opts.command:
        case "rag":
            rag(searcher, rag_client, cli_opts.query)
        case _:
            parser.print_help()
