import string
from functools import reduce
from typing import Callable

TextProcFunc = Callable[[str], str] | Callable[[str], list[str]]


def tokenize(text: str) -> list[str]:
    text_splits = text.split()
    return [text for text in text_splits if text != ""]


def remove_punctuation(text: str) -> str:
    str_trans_map = str.maketrans("", "", string.punctuation)
    return text.translate(str_trans_map)


def convert_to_lower(text: str) -> str:
    return text.lower()


def clean_text(text: str) -> list[str]:
    func_list: list[TextProcFunc] = [
        convert_to_lower,
        remove_punctuation,
        tokenize,
    ]
    return reduce(lambda acc, func: func(acc), func_list, text)
