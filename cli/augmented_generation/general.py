from augmented_generation.augmented_generation import LLMOutputType, LLMSummarizer
from augmented_generation.opts import get_opts
from hybrid_search.hybrid_search import HybridSearch

RESULT_PADDING = 2


def run(api_key: str) -> None:
    cli_opts, parser = get_opts()

    searcher = HybridSearch()
    rag_client = LLMSummarizer(api_key=api_key)

    query = cli_opts.query
    results = searcher.rrf_search(query, limit=cli_opts.limit)

    match cli_opts.command:
        case "rag":
            rag_response = rag_client.answer(query, results, LLMOutputType.BASIC)
            rag_response_header = "RAG Response"
        case "summarize":
            rag_response = rag_client.answer(
                query, results, LLMOutputType.COMPREHENSIVE
            )
            rag_response_header = "LLM Summary"
        case "citations":
            rag_response = rag_client.answer(query, results, LLMOutputType.CITATIONS)
            rag_response_header = "LLM Answer"
        case "question":
            rag_response = rag_client.answer(query, results, LLMOutputType.QUESTION)
            rag_response_header = "Answer"
        case _:
            parser.print_help()

    print("Search Results:")
    for result in results:
        print(f"{' ':<{RESULT_PADDING}}- {result['title']}")

    print(f"\n{rag_response_header}:\n\n{rag_response}")
