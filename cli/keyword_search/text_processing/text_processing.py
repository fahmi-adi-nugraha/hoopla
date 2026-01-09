import string
from collections.abc import Callable
from pathlib import Path
from typing import Iterable, TypeVar

from nltk.stem import PorterStemmer

STOPWORDS_FILE = Path("data/stopwords.txt")


class TextProcessingContext:
    def __init__(self, stemmer: PorterStemmer | None = None):
        with open(STOPWORDS_FILE, "r") as swf:
            self.stopwords = swf.read().splitlines()

        if stemmer is None:
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = stemmer


T = TypeVar("T")
PipelineStep = Callable[[T, TextProcessingContext], T]


def _run_pipeline(
    data: T, ctx: TextProcessingContext, steps: Iterable[PipelineStep[T]]
) -> T:
    for step in steps:
        data = step(data, ctx)
    return data


def convert_to_lower(text: str, ctx: TextProcessingContext | None = None) -> str:
    return text.lower()


def remove_punctuation(text: str, ctx: TextProcessingContext | None = None) -> str:
    str_trans_map = str.maketrans("", "", string.punctuation)
    return text.translate(str_trans_map)


def tokenize(text: str, ctx: TextProcessingContext | None = None) -> list[str]:
    text_splits = text.split()
    return [text for text in text_splits if text != ""]


def stem_tokens(tokens: list[str], ctx: TextProcessingContext) -> list[str]:
    return [ctx.stemmer.stem(token) for token in tokens]


def remove_stop_words(tokens: list[str], ctx: TextProcessingContext) -> list[str]:
    return [token for token in tokens if token not in ctx.stopwords]


def clean_text(text: str, ctx: TextProcessingContext) -> list[str]:
    steps = [
        convert_to_lower,
        remove_punctuation,
        tokenize,
        remove_stop_words,
        stem_tokens,
    ]
    return _run_pipeline(text, ctx, steps)
