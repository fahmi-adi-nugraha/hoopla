import string
from functools import partial, reduce
from typing import Callable

TextProcFunc = Callable[[str | list[str]], str | list[str]]


def remove_stop_words(text: list[str], stop_words: list[str]) -> list[str]:
    return [word for word in text if word not in stop_words]


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
    ]
    return reduce(lambda acc, func: func(acc), func_list, text)
