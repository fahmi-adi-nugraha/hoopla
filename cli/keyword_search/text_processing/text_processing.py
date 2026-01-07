import string
from pathlib import Path
from typing import Callable

from nltk.stem import PorterStemmer

STOPWORDS_FILE = Path("data/stopwords.txt")

# TextProcFunc = Callable[[str | list[str]], str | list[str]]
# TextProcFunc = (
#     Callable[[str], str]
#     | Callable[[list[str]], list[str]]
#     | Callable[[str], list[str]]
#     | Callable[[list[str]], list[str]]
# )


def tokenize(text: str) -> list[str]:
    text_splits = text.split()
    return [text for text in text_splits if text != ""]


class TextClean:
    # def __init__(self, tokens: str | list[str]) -> None:
    def __init__(self, stemmer: PorterStemmer | None = None) -> None:
        self.stopwords: list[str] = []
        with open(STOPWORDS_FILE, "r") as swf:
            self.stopwords = swf.read().splitlines()

        self.stemmer = stemmer if stemmer is not None else PorterStemmer()
        self.result: str | list[str] = None

    def stem_tokens(self) -> "TextClean":
        self.result = [self.stemmer.stem(token) for token in self.result]
        return self

    def remove_stop_words(self) -> "TextClean":
        self.result = [token for token in self.result if token not in self.stopwords]
        return self

    def tokenize(self) -> "TextClean":
        self.result = tokenize(self.result)
        return self

    def remove_punctuation(self) -> "TextClean":
        str_trans_map = str.maketrans("", "", string.punctuation)
        self.result = self.result.translate(str_trans_map)
        return self

    def convert_to_lower(self) -> "TextClean":
        self.result = self.result.lower()
        return self

    # def clean_text(self, text: str) -> list[str]:
    #     func_list: list[TextProcFunc]
    #     if stopwords is None:
    #         func_list = [
    #             convert_to_lower,
    #             partial(remove_punctuation, stopwords=stopwords),
    #             tokenize,
    #             remove_stop_words,
    #             stem_tokens,
    #         ]
    #     else:
    #         func_list = [
    #             convert_to_lower,
    #             partial(remove_punctuation, stopwords=stopwords),
    #             tokenize,
    #             remove_stop_words,
    #             stem_tokens,
    #         ]
    #
    #     return reduce(lambda acc, func: func(acc), func_list, text)

    # def clean_text_up_to_tokenize(self, text: str) -> list[str]:
    #     func_list: list[TextProcFunc] = [
    #         convert_to_lower,
    #         remove_punctuation,
    #         tokenize,
    #     ]
    #     return reduce(lambda acc, func: func(acc), func_list, text)

    # def clean_text_finish(self, text: list[str], stopwords: list[str]) -> list[str]:
    #     func_list: list[TextProcFunc] = [
    #         partial(remove_stop_words, stopwords=stopwords),
    #         stem_tokens,
    #     ]
    #     return reduce(lambda acc, func: func(acc), func_list, text)
