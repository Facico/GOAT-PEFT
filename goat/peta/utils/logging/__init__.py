import logging
from typing import Any, Callable, Optional

from .rich import *


def titled_log(
    title: str,
    msg: Any,
    title_width: int = 50,
    log_fn: Callable = print,
):
    log_fn(f"{title:=^{title_width}}")
    log_fn(msg)
    log_fn(f"{'':=^{title_width}}")


class TitledLog:
    """
    Examples:
        >>> with TitledLog('msg'):
        >>>     ... # do_something
    """

    def __init__(
        self,
        title: str = None,
        title_width: int = 50,
        log_fn=print,
        log_kwargs: Optional[dict] = None,
    ):
        """

        Args:
            description (str, optional): _description_. Defaults to None.
            logger (logging.Logger, optional): _description_. Defaults to log.info.
        """
        self.title = title if title is not None else ""
        self.title_width = title_width

        self.log_fn = log_fn
        # log_kwargs
        if self.log_fn == print:
            self.log_kwargs = dict()
        else:
            self.log_kwargs = dict(stacklevel=2)
        if log_kwargs is not None:
            self.log_kwargs.update(log_kwargs)

    def __enter__(self):
        self.log_fn(
            f"{self.title:=^{self.title_width}}",
            **self.log_kwargs,
        )

    def __exit__(self, exc_type, exc_value, tb):
        self.log_fn(
            f"{'':=^{self.title_width}}",
            **self.log_kwargs,
        )
