from typing import Any

from google import genai

RAG_MODEL = "gemini-2.5-flash"


class RAG:
    def __init__(self, api_key: str, model_name: str = "") -> None:
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = RAG_MODEL

        self.client = genai.Client(api_key=api_key)

    def answer(self, query: str, documents: list[dict[str, Any]]) -> str:
        documents_string = "#~#".join(
            [
                f"{document['title']} - {document['description']}"
                for document in documents
            ]
        )
        prompt = f"""Answer the question or provide information based on the provided
        documents. This should be tailored to Hoopla users. Hoopla is a movie streaming
        service.

        Query: {query}

        Documents:
        {documents_string}

        The documents are contained in a string with the following format:
        - Each entry consists of the title of the movie and the description separated by
          a hyphen
        - Each entry is separated by this sequence of characters: #~#
        
        Provide a comprehensive answer that addresses the query:"""

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )

        return response.text
