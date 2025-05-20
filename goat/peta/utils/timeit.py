import logging
import time
from typing import Optional

_log = logging.getLogger(__name__)


class TimeIt:
    """
    Examples:
        >>> with TimeIt('msg'):
        >>>     ... # do_something
    """

    def __init__(self, description: str = None, logger=print):
        """

        Args:
            description (str, optional): _description_. Defaults to None.
            logger (logging.Logger, optional): _description_. Defaults to log.info.
        """
        self.logger = logger
        self.description = description if description is not None else "timeit"
        if self.logger == print:
            self.logger_kwargs = dict()
        else:
            self.logger_kwargs = dict(stacklevel=2)

    def __enter__(self):
        self.start = time.time()
        self.logger(f"[start] {self.description}", **self.logger_kwargs)

    def __exit__(self, exc_type, exc_value, tb):
        self.logger(
            f"[end] {self.description}: {(time.time()-self.start):.2f}s",
            **self.logger_kwargs,
        )
