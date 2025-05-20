r"""
This module provides utilities for logging with rich output formatting.

It defines a function `setup_colorlogging` that sets up a logging configuration
with a `RichHandler` that outputs log messages with rich formatting to the console.
It also defines two functions `pprint_code` and `pprint_yaml` that pretty print
code snippets in different formats using the `Syntax` class from the `rich.syntax`
module.

Example usage:

    >>> from utils.logging.rich import setup_colorlogging, pprint_yaml

    >>> setup_colorlogging()
    >>> pprint_yaml('foo: bar')

"""
import logging
from functools import partial
from logging import getLogger

from rich.console import Console
from rich.logging import RichHandler
from rich.syntax import Syntax

log = getLogger(__name__)
_console = Console()

__all__ = [
    "setup_colorlogging",
    "pprint_code",
    "pprint_yaml",
    "pprint_json",
]


def setup_colorlogging(force=False, **config_kwargs):
    FORMAT = "%(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler()],
        force=force,
        **config_kwargs,
    )


setup_colorlogging()


def pprint_code(code: str, format: str):
    """pretty print code

    Args:
        code (str): code str
        format (str): code format, for example 'yaml'
    """
    s = Syntax(code, format)
    _console.print(s)


pprint_yaml = partial(pprint_code, format="yaml")
"""pretty print yaml code"""
pprint_json = partial(pprint_code, format="json")
"""pretty print json code"""
