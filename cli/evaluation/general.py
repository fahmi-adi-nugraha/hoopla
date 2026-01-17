import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

from hybrid_search.hybrid_search import HybridSearch

GOLDEN_DATASET_PATH = "data/golden_dataset.json"


def _load_golden_dataset() -> dict[str, list[dict[str, str | list[str]]]]:
    with open(Path(GOLDEN_DATASET_PATH), "r") as golden:
        return json.load(golden)


def run(cli_opts: Namespace, parser: ArgumentParser) -> None:
    golden_dataset = _load_golden_dataset()

    hybrid_searcher = HybridSearch()

    padding = 4
    limit = cli_opts.limit
    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        results = hybrid_searcher.rrf_search(query, limit=limit)
        movies_retrieved = [result["title"] for result in results]
        movies_retrieved_relevant = [
            movie for movie in movies_retrieved if movie in test_case["relevant_docs"]
        ]

        precision = len(movies_retrieved_relevant) / len(movies_retrieved)
        recall = len(movies_retrieved_relevant) / len(test_case["relevant_docs"])

        print(f"k={limit}\n")
        print(f"{'-':<{padding}}Query: {query}")
        print(f"{' ':<{padding}}- Precision@{limit}: {precision:.4f}")
        print(f"{' ':<{padding}}- Recall@{limit}: {recall:.4f}")
        print(f"{' ':<{padding}}- Retrieved: {', '.join(movies_retrieved)}")
        print(f"{' ':<{padding}}- Relevant: {', '.join(movies_retrieved_relevant)}")
