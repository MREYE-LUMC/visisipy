from __future__ import annotations

from typing import Any


def build_args(*, non_null_defaults: set[str], **kwargs) -> dict[str, Any]:
    """Build a dictionary of arguments for the analysis.

    Parameters
    ----------
    non_null_defaults : set[str], optional
        A list of argument names that are optional but do not have `None` as a default value.
    **kwargs : Any
        The arguments to be included in the dictionary.

    Returns
    -------
    dict[str, Any]
        A dictionary of arguments for the analysis.
    """
    null_defaults = kwargs.keys() - non_null_defaults

    result = {}

    for key in null_defaults:
        result[key] = kwargs[key]

    for key in non_null_defaults:
        if key in kwargs and kwargs[key] is not None:
            result[key] = kwargs[key]

    return result
