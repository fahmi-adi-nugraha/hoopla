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

    def enhance(self, query: str, enhancement_type: str | None = None) -> str:
        # Need to add error handling later
        match enhancement_type:
            case "spell":
                query = self.__enhance_spelling(query)
            case _:
                pass

        return query
