r"""
This module contains miscellaneous useful functions and classes.

Submodules:
- args: Contains functions for parsing command-line arguments.
- collections: Contains additional collection classes.
- logging: Contains additional logging classes and functions.
- path: Contains functions for working with file paths.
- timeit: Contains functions for timing code execution.
"""
from typing import Iterable
import importlib

from .args import *
from .logging import TitledLog, titled_log
from .timeit import *


def first(iterable: Iterable):
    R"""Returns the first element of `iterable`.

    Examples:

        >>> first_batch = first(data_loader)

    Args:
        iterable (Iterable): _description_

    Returns:
        Any: _description_
    """
    return next(iter(iterable))


def import_object(module_name: str, object_name: str):
    module = importlib.import_module(module_name)
    obj = getattr(module, object_name)
    return obj
