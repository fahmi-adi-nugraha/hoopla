import logging
from logging import Logger


def new_logger() -> Logger:
    logger = logging.getLogger(__name__)

    # Set log level
    logger.setLevel(logging.DEBUG)

    # Configure file handler
    file_handler = logging.FileHandler("rrf_search.log")
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
