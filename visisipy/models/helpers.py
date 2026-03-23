"""Helper functions for eye models."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T", bound=type)


def _collect_subclasses(cls: T, registry: dict[str, T]) -> None:
    registry[cls.__name__] = cls
    for subclass in cls.__subclasses__():
        _collect_subclasses(subclass, registry)
