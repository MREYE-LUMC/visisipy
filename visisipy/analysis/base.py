from __future__ import annotations

import inspect
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from visisipy.backend import BaseBackend, get_backend

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
    parameter_names = list(signature.parameters)

    if parameter_names[0] != "model":
        raise ValueError("The first parameter of an analysis function must be 'model'.")

    if signature.parameters["model"].annotation != "EyeModel | None":
        raise ValueError(
            f"The 'model' parameter of an analysis function must have type 'EyeModel | None', "
            f"got '{signature.parameters['model'].annotation}'"
        )

    if "backend" not in parameter_names:
        raise ValueError(
            "The analysis function must have a keyword-only 'backend' parameter of type 'type["
            "BaseBackend]'.")

    if signature.parameters["backend"].kind.name != "KEYWORD_ONLY":
        raise ValueError("The 'backend' parameter of an analysis function must be keyword-only.")

    if signature.parameters["backend"].annotation != "type[BaseBackend]":
        raise ValueError(
            f"The 'backend' parameter of an analysis function must have type 'type[BaseBackend]', "
            f"got '{signature.parameters['backend'].annotation}'"
        )

    if "return_raw_result" not in parameter_names:
        raise ValueError(
            "The analysis function must have a keyword-only 'return_raw_result' parameter of type 'bool'.")

    if signature.parameters["return_raw_result"].kind.name != "KEYWORD_ONLY":
        raise ValueError("The 'return_raw_result' parameter of an analysis function must be keyword-only.")

    if signature.parameters["return_raw_result"].annotation != "bool":
        raise ValueError(
            f"The 'return_raw_result' parameter of an analysis function must have type 'bool', "
            f"got '{signature.parameters['return_raw_result'].annotation}'"
        )


def _build_model(model: EyeModel, backend: type[BaseBackend]) -> None:
    """
    Build the model in OpticStudio if it is not already built.

    Parameters
    ----------
    model : Any
        The model to build.
    """
    if backend.model is None or backend.model.eye_model is not model:
        backend.build_model(model)


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def analysis(function: Callable[..., tuple[T1, T2]]) -> Callable:
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
            backend: type[BaseBackend] | None = None,
            **kwargs: Any,
    ) -> T1 | tuple[T1, T2]:
        if backend is None:
            backend = get_backend()

        if model is not None:
            _build_model(model, backend)

        result, raw_result = function(model, *args, return_raw_result=return_raw_result, backend=backend, **kwargs)

        if return_raw_result:
            return result, raw_result

        return result

    return analysis_wrapper
