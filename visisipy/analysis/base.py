from __future__ import annotations

import inspect
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

from visisipy.backend import get_backend

if TYPE_CHECKING:
    from visisipy.models import EyeModel


def _validate_analysis_signature(function: Callable[..., tuple[Any, Any]]) -> None:
    """
    Validate the signature of an analysis function.

    The first parameter must be 'model' with type 'EyeModel | None'. The last parameter must be 'return_raw_result'
    with type 'bool'.

    Parameters
    ----------
    function : Callable
        The analysis function to validate.

    Raises
    ------
    ValueError
        If the signature of the analysis function is invalid.
    """

    signature = inspect.signature(function)
    first_parameter, *_, last_parameter = signature.parameters

    if first_parameter != "model":
        raise ValueError("The first parameter of an analysis function must be 'model'.")

    if signature.parameters["model"].annotation != "EyeModel | None":
        raise ValueError(
            f"The first parameter of an analysis function must have type 'EyeModel | None', "
            f"got '{signature.parameters['model'].annotation}'"
        )

    if last_parameter != "return_raw_result":
        raise ValueError("The last parameter of an analysis function must be 'return_raw_result'.")

    if signature.parameters["return_raw_result"].annotation != "bool":
        raise ValueError(
            f"The last parameter of an analysis function must have type 'bool', "
            f"got '{signature.parameters['return_raw_result'].annotation}'"
        )


def _build_model(model: EyeModel) -> None:
    """
    Build the model in OpticStudio if it is not already built.

    Parameters
    ----------
    model : Any
        The model to build.
    """
    backend = get_backend()

    if backend.model is None or backend.model.eye_model is not model:
        backend.build_model(model)


def analysis(function: Callable[..., tuple[Any, Any]]) -> Callable:
    """
    Decorator for analysis functions.

    This decorator is used to mark a function as an analysis function. Analysis functions are used to perform various
    analyses on the optical system.

    Parameters
    ----------
    function : Callable
        The analysis function.

    Returns
    -------
    Callable
        The analysis function.
    """
    _validate_analysis_signature(function)

    @wraps(function)
    def analysis_wrapper(
        model: EyeModel | None = None,
        *args: Any,
        return_raw_result: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        if model is not None:
            _build_model(model)

        result, raw_result = function(model, *args, return_raw_result=return_raw_result, **kwargs)

        if return_raw_result:
            return result, raw_result

        return result

    return analysis_wrapper
