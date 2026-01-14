from google import genai

ENHANCER_MODEL = "gemini-2.5-flash"


class QueryEnhancer:
    def __init__(self, api_key: str, model_name: str = "") -> None:
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = ENHANCER_MODEL
        self.client = genai.Client(api_key=api_key)

    def __enhance_spelling(self, query: str) -> str:
        prompt = f"""Fix any spelling errors in this movie search query.

        Only correct obvious typos. Don't change correctly spelled words.
        
        Query: "{query}"

        If no errors, return the original query.

        Make sure that regardless of whether or not the query was corrected the final
        result does not have any quotes around it and is lowercase.

        Corrected:"""

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )

        return response.text

    def __enhance_by_rewriting(self, query: str) -> str:
        prompt = f"""Rewrite this movie search query to be more specific and searchable.

        
        Original: "{query}"

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep it concise (under 10 words)
        - It should be a google style search query that's very specific
        - Don't use boolean logic

        Examples:

        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        Make sure that final result does not have any quotes around it.

        Rewritten query:"""

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )

        return response.text

    def enhance(self, query: str, enhancement_type: str | None = None) -> str:
        # Need to add error handling later
        match enhancement_type:
            case "spell":
                query = self.__enhance_spelling(query)
            case "rewrite":
                query = self.__enhance_by_rewriting(query)
            case _:
                pass

        return query
