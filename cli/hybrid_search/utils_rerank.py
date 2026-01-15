from time import sleep

from google import genai

RERANKER_MODEL = "gemini-2.5-flash"
RERANKER_SLEEP_LENGTH_SECONDS = 3


class LLMReranker:
    def __init__(self, api_key: str, model_name: str = "") -> None:
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = RERANKER_MODEL
        self.client = genai.Client(api_key=api_key)

    def __rerank_individually(
        self, query: str, results: list[dict[str, str | int | float]]
    ) -> list[dict[str, str | int | float]]:
        results_reranked: list[dict[str, str | int | float]] = []
        for result in results:
            prompt = f"""Rate how well this movie matches the search query.
            
            Query: "{query}"
            Movie: {result.get("title", "")} - {result.get("description", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate the match between the movie and the search query by giving a score
            between 0-10 (10 = perfect match). Give me ONLY the number of the score in
            your response, no other text or explanation.

            Score:"""

            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt
            )

            # print(f"LLM response: {response.text}")

            result["rerank_score"] = float(response.text)

            results_reranked.append(result)
            sleep(RERANKER_SLEEP_LENGTH_SECONDS)

        return sorted(results_reranked, key=lambda x: x["rerank_score"], reverse=True)

    def rerank(
        self,
        query: str,
        results: list[dict[str, str | int | float]],
        rerank_method: str | None,
    ) -> list[dict[str, str | int | float]]:
        results_reranked: list[dict[str, str | int | float]]
        match rerank_method:
            case "individual":
                results_reranked = self.__rerank_individually(query, results)
            case _:
                pass

        return results_reranked
