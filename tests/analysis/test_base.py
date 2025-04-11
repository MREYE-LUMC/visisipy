from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from visisipy import EyeModel
from visisipy.analysis import base

if TYPE_CHECKING:
    from visisipy.backend import BaseBackend


class MockBackend:
    def __init__(self):
        self.model = None

    def build_model(self, model):
        self.model = SimpleNamespace(eye_model=model)


@pytest.fixture
def mock_backend(monkeypatch):
    """Mock the backend for testing."""

    backend = MockBackend()

    monkeypatch.setattr(base, "get_backend", lambda: backend)

    return backend


@pytest.fixture
def example_analysis():
    """Example analysis function to test the analysis decorator."""

    def example_analysis(model: EyeModel | None, x: int, *, return_raw_result: bool, backend: type[BaseBackend]):
        return x, x

    return example_analysis


class TestAnalysisDecorator:
    def test_decorator(self, mock_backend, example_analysis):
        decorated_analysis = base.analysis(example_analysis)

        assert decorated_analysis(EyeModel(), 1) == 1
        assert decorated_analysis(EyeModel(), 1, return_raw_result=True) == (1, 1)

    def test_build_model(self, mock_backend, example_analysis):
        model = EyeModel()

        decorated_analysis = base.analysis(example_analysis)
        decorated_analysis(model, 1)

        assert mock_backend.model is not None
        assert mock_backend.model.eye_model == model

    def test_get_backend(self, mock_backend):
        def example_analysis(
            model: EyeModel | None, x: int, *, return_raw_result: bool, backend: type[BaseBackend]
        ) -> tuple[type[BaseBackend], int]:
            return backend, x

        decorated_analysis = base.analysis(example_analysis)

        assert decorated_analysis(EyeModel(), 1) == mock_backend

    def test_pass_backend(self):
        def example_analysis(
            model: EyeModel | None, x: int, *, return_raw_result: bool, backend: type[BaseBackend]
        ) -> tuple[type[BaseBackend], int]:
            return backend, x

        decorated_analysis = base.analysis(example_analysis)
        backend = MockBackend()

        assert decorated_analysis(EyeModel(), 1, backend=backend) == backend

    def test_no_model_raises_valueerror(self):
        def example_analysis(x, *, return_raw_result: bool = False, backend: type[BaseBackend]):
            return x, x

        with pytest.raises(
            ValueError,
            match="The first parameter of an analysis function must be 'model'",
        ):
            base.analysis(example_analysis)

    def test_invalid_model_annotation_raises_valueerror(self):
        def example_analysis(model: str, x, *, return_raw_result: bool = False, backend: type[BaseBackend]):
            return x, x

        with pytest.raises(
            ValueError,
            match="The 'model' parameter of an analysis function must have type 'EyeModel | None', got 'str'",
        ):
            base.analysis(example_analysis)

    def test_no_return_raw_result_raises_valueerror(self):
        def example_analysis(model: EyeModel | None, x, *, backend: type[BaseBackend]):
            return x, x

        with pytest.raises(
            ValueError,
            match="The analysis function must have a keyword-only 'return_raw_result' parameter of type 'bool'",
        ):
            base.analysis(example_analysis)

    def test_no_keyword_only_return_raw_result_raises_valueerror(self):
        def example_analysis(
            model: EyeModel | None,
            x,
            return_raw_result: bool,
            *,
            backend: type[BaseBackend],
        ):
            return x, x

        with pytest.raises(
            ValueError,
            match="The 'return_raw_result' parameter of an analysis function must be keyword-only",
        ):
            base.analysis(example_analysis)

    def test_invalid_return_raw_result_annotation_raises_valueerror(self):
        def example_analysis(model: EyeModel | None, x, *, return_raw_result: str, backend: type[BaseBackend]):
            return x, x

        with pytest.raises(
            ValueError,
            match="The 'return_raw_result' parameter of an analysis function must have type 'bool', got 'str'",
        ):
            base.analysis(example_analysis)

    def test_no_backend_raises_valueerror(self):
        def example_analysis(model: EyeModel | None, x, *, return_raw_result: bool):
            return x, x

        with pytest.raises(
            ValueError,
            match=r"The analysis function must have a keyword-only 'backend' parameter of type 'type\[BaseBackend\]'",
        ):
            base.analysis(example_analysis)

    def test_no_keyword_only_backend_raises_valueerror(self):
        def example_analysis(model: EyeModel | None, x, backend: type[BaseBackend], *, return_raw_result: bool):
            return x, x

        with pytest.raises(
            ValueError,
            match=r"The 'backend' parameter of an analysis function must be keyword-only",
        ):
            base.analysis(example_analysis)

    def test_invalid_backend_annotation_raises_valueerror(self):
        def example_analysis(model: EyeModel | None, x, *, return_raw_result: bool, backend: str):
            return x, x

        with pytest.raises(
            ValueError,
            match=r"The 'backend' parameter of an analysis function must have type 'type\[BaseBackend\]', "
            r"got 'str'",
        ):
            base.analysis(example_analysis)
