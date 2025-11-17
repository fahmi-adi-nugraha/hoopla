import string
from functools import partial, reduce
from typing import Callable

from nltk.stem import PorterStemmer

# TextProcFunc = Callable[[str | list[str]], str | list[str]]
TextProcFunc = (
    Callable[[str], str]
    | Callable[[list[str]], list[str]]
    | Callable[[str], list[str]]
    | Callable[[list[str]], list[str]]
)


def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def remove_stop_words(tokens: list[str], stop_words: list[str]) -> list[str]:
    return [token for token in tokens if token not in stop_words]


def tokenize(text: str) -> list[str]:
    text_splits = text.split()
    return [text for text in text_splits if text != ""]


def remove_punctuation(text: str) -> str:
    str_trans_map = str.maketrans("", "", string.punctuation)
    return text.translate(str_trans_map)


def convert_to_lower(text: str) -> str:
    return text.lower()


def clean_text(text: str, stop_words: list[str]) -> list[str]:
    func_list: list[TextProcFunc] = [
        convert_to_lower,
        remove_punctuation,
        tokenize,
        partial(remove_stop_words, stop_words=stop_words),
        stem_tokens,
    ]
    return reduce(lambda acc, func: func(acc), func_list, text)


def clean_text_up_to_tokenize(text: str) -> list[str]:
    func_list: list[TextProcFunc] = [
        convert_to_lower,
        remove_punctuation,
        tokenize,
    ]
    return reduce(lambda acc, func: func(acc), func_list, text)


def clean_text_finish(text: list[str], stop_words: list[str]) -> list[str]:
    func_list: list[TextProcFunc] = [
        partial(remove_stop_words, stop_words=stop_words),
        stem_tokens,
    ]
    return reduce(lambda acc, func: func(acc), func_list, text)
