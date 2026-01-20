from enum import Enum
from typing import Any

from google import genai

RAG_MODEL = "gemini-2.5-flash"


class LLMOutputType(Enum):
    BASIC = 1
    COMPREHENSIVE = 2
    CITATIONS = 3
    QUESTION = 4


class LLMSummarizer:
    def __init__(self, api_key: str, model_name: str = "") -> None:
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = RAG_MODEL

        self.client = genai.Client(api_key=api_key)

    def __get_prompt(
        self, query: str, documents: list[dict[str, Any]], output_type: LLMOutputType
    ) -> str:
        documents_string = "#~#".join(
            [
                f"{document['title']} - {document['description']}"
                for document in documents
            ]
        )

        prompt = ""
        match output_type:
            case LLMOutputType.BASIC:
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

            case LLMOutputType.COMPREHENSIVE:
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

            case LLMOutputType.CITATIONS:
                prompt = f"""Answer the question or provide information based on the
                provided documents.

                This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                If not enough information is available to give a good answer, say so but
                give as good of an answer as you can while citing the sources you have.

                Query: {query}

                Search Results:
                {documents_string}

                The search results are contained in a string with the following format:
                - Each entry consists of the title of the movie and the description separated by
                  a hyphen
                - Each entry is separated by this sequence of characters: #~#

                Instructions:
                - Provide a comprehensive answer that addresses the query
                - Cite sources using [1], [2], etc. format when referencing information
                - If sources disagree, mention the different viewpoints
                - If the answer isn't in the documents, say "I don't have enough
                  information"
                - Be direct and informative
                
                Answer:"""

            case LLMOutputType.QUESTION:
                prompt = f"""Answer the user's question based on the provided movies
                that are available on Hoopla.

                This should be tailored to Hoopla users. Hoopla is a movie streaming
                service.

                Question: {query}

                Documents:
                {documents_string}

                The search results are contained in a string with the following format:
                - Each entry consists of the title of the movie and the description separated by
                  a hyphen
                - Each entry is separated by this sequence of characters: #~#

                Instructions:
                - Answer questions directly and concisely
                - Be casual and conversational
                - Don't be cringe or hype-y
                - Talk like a normal person would in a chat conversation
                
                Answer:"""

            case _:
                pass

        return prompt

    def answer(
        self, query: str, documents: list[dict[str, Any]], answer_type: LLMOutputType
    ) -> str:
        prompt = self.__get_prompt(query, documents, answer_type)

        if prompt is None or not prompt:
            return ""

        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt
        )

        return response.text
