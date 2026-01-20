from enum import Enum
from typing import Any

from google import genai

RAG_MODEL = "gemini-2.5-flash"


class RAGAnswerType(Enum):
    BASIC = 1
    COMPREHENSIVE = 2


class RAG:
    def __init__(self, api_key: str, model_name: str = "") -> None:
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = RAG_MODEL

        self.client = genai.Client(api_key=api_key)

    def __summary_basic(self, query: str, documents: list[dict[str, Any]]) -> str:
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

    def __summary_comprehensive(
        self, query: str, documents: list[dict[str, Any]]
    ) -> str:
        documents_string = "#~#".join(
            [
                f"{document['title']} - {document['description']}"
                for document in documents
            ]
        )
        prompt = f"""Provide information useful to this query by synthesizing
        information from multiple search results in detail. The goal is to provide
        comprehensive information so that users know what their options are. Your
        response should be information-dense and concise, with several key pieces of
        infromation about the genre, plot, etc. of each movie.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Search Results:
        {documents_string}

        The search results are contained in a string with the following format:
        - Each entry consists of the title of the movie and the description separated by
          a hyphen
        - Each entry is separated by this sequence of characters: #~#
        
        Provide a comprehensive 3-4 sentence answer that combines information from
        multiple sources:"""

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )

        return response.text

    def answer(
        self, query: str, documents: list[dict[str, Any]], answer_type: RAGAnswerType
    ) -> str:
        response = ""
        match answer_type:
            case RAGAnswerType.BASIC:
                response = self.__summary_basic(query, documents)
            case RAGAnswerType.COMPREHENSIVE:
                response = self.__summary_comprehensive(query, documents)
            case _:
                pass
        return response
