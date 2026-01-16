import json
from time import sleep

from google import genai
from sentence_transformers.cross_encoder import CrossEncoder

RERANKER_MODEL = "gemini-2.5-flash"
RERANKER_SLEEP_LENGTH_SECONDS = 3
RERANKER_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"


class LLMReranker:
    def __init__(self, api_key: str, model_name: str = "") -> None:
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = RERANKER_MODEL
        self.client = genai.Client(api_key=api_key)
        self.cross_encoder = CrossEncoder(RERANKER_CROSS_ENCODER_MODEL)

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

            result["rerank_score"] = float(response.text)

            results_reranked.append(result)
            sleep(RERANKER_SLEEP_LENGTH_SECONDS)

        return sorted(results_reranked, key=lambda x: x["rerank_score"], reverse=True)

    def __rerank_batch(
        self, query: str, results: list[dict[str, str | int | float]]
    ) -> list[dict[str, str | int | float]]:
        prompt = f"""Rank these movies by relevance to the search query.
        
        Query: "{query}"

        Movies:
        {results}

        The movies are presented as the string representation of a Python list of
        dictionaries where each dictionary contains information about a movie. The
        dictionaries have 'id' field containing the ID of the movie and a 'description'
        field containing a synopsis of the movie. Use the synopsis to determine how
        relevant each movie is to the search query. Return ONLY the IDs in order of
        relevance (best match first). Return an valid JSON list, nothing else. For
        example:

        [75, 12, 34, 2, 1]
        """

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )

        ranked_ids = json.loads(response.text)
        results_reranked: list[dict[str, str | int | float]] = []
        for i, movie_id in enumerate(ranked_ids):
            for result in results:
                if result["id"] == movie_id:
                    result["rerank_rank"] = i + 1
                    results_reranked.append(result)

        return results_reranked

    def __rerank_cross_encoder(
        self, query: str, results: list[dict[str, str | int | float]]
    ) -> list[dict[str, str | int | float]]:
        pairs: list[tuple[str, str]] = []
        for result in results:
            pairs.append(
                (query, f"{result.get('title', '')} - {result.get('document', '')}")
            )
        scores = self.cross_encoder.predict(pairs, show_progress_bar=True)
        for score, result in zip(scores, results):
            result["cross_encoder_score"] = score
        return sorted(results, key=lambda x: x["cross_encoder_score"], reverse=True)

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
            case "batch":
                results_reranked = self.__rerank_batch(query, results)
            case "cross_encoder":
                results_reranked = self.__rerank_cross_encoder(query, results)
            case _:
                pass

        return results_reranked
