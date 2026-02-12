"""Generate test data for analysis results tests.

This script runs the analyses defined in `config.py` and saves the results to CSV files in
the `tests/_data/analysis_results` directory. The test data is used to verify if the analysis
results are consistent across different versions of visisipy and the backends. The result tests
are explicitly not intended to serve as a comparison between the different backends.

The script can be run with the following command:

.. code-block:: bash
    hatch run generate-test-data

The tests to be generated are defined in the `TESTS` dictionary in `config.py`.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING
from warnings import filterwarnings

from config import (
    OPTICSTUDIO_BACKEND_SETTINGS,
    OPTILAND_BACKEND_SETTINGS,
    TEST_DATA_DIR,
    TESTS,
)
from loguru import logger

from visisipy.models.base import EyeModel
from visisipy.models.geometry import BiconicSurface, EyeGeometry, StandardSurface, Stop
from visisipy.models.materials import NavarroMaterials543
from visisipy.opticstudio.backend import OpticStudioBackend
from visisipy.optiland.backend import OptilandBackend

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from visisipy.backend import BackendSettings, BaseBackend


def model() -> EyeModel:
    """Create the eye model used for generating the test data."""
    materials = NavarroMaterials543()
    geometry = EyeGeometry(
        cornea_front=BiconicSurface(
            radius=7.6967, asphericity=-0.2304, thickness=0.5615, radius_x=7.9487, asphericity_x=-0.2304
        ),
        cornea_back=StandardSurface(radius=6.2343, asphericity=-0.1444, thickness=3.345),
        pupil=Stop(semi_diameter=1.0),
        lens_front=StandardSurface(radius=10.2, asphericity=-3.1316, thickness=3.17),
        lens_back=StandardSurface(radius=-5.4537, asphericity=0, thickness=17.2285),
        retina=StandardSurface(radius=-12.5000, asphericity=0.033),
    )
    return EyeModel(geometry=geometry, materials=materials)


def initialize_backends() -> list[type[BaseBackend]]:
    """Initialize the backends with the default settings for generating the test data."""
    OpticStudioBackend.initialize(**OPTICSTUDIO_BACKEND_SETTINGS)
    OptilandBackend.initialize(**OPTILAND_BACKEND_SETTINGS)

    return [OpticStudioBackend, OptilandBackend]


_BACKEND_SETTINGS: dict[type[BaseBackend], BackendSettings] = {
    OpticStudioBackend: OPTICSTUDIO_BACKEND_SETTINGS,
    OptilandBackend: OPTILAND_BACKEND_SETTINGS,
}


def reset_backend_settings(backend: type[BaseBackend]) -> None:
    """Reset the backend settings to the default values for generating the test data."""
    if backend not in _BACKEND_SETTINGS:
        raise ValueError(f"Unknown backend type: {backend}")

    settings = _BACKEND_SETTINGS[backend]
    backend.update_settings(**settings)


def get_file_name(test_name: str, backend_name: str, **extra_settings: str) -> Path:
    """Generate a file name for the test data based on the test name, backend name, and extra settings."""
    file_name = f"{test_name}_{backend_name}"

    if extra_settings:
        settings_str = "_".join(f"{key}-{value}" for key, value in extra_settings.items())
        file_name += f"_{settings_str}"

    return TEST_DATA_DIR / f"{file_name}.csv"


_MODEL_SUFFIXES: dict[type[BaseBackend], str] = {
    OpticStudioBackend: ".zmx",
    OptilandBackend: ".json",
}


def build_and_save_model(model: EyeModel, path: Path, backends: Iterable[type[BaseBackend]]) -> None:
    for backend in backends:
        suffix: str

        if backend not in _MODEL_SUFFIXES:
            raise ValueError(f"Unknown backend type: {backend}")

        suffix = _MODEL_SUFFIXES[backend]

        reset_backend_settings(backend)

        model_path = path.with_suffix(suffix)

        logger.info("Saving model for backend {} to {}...", backend.type, model_path)
        backend.build_model(model)
        backend.save_model(model_path)


def main(args: argparse.Namespace) -> None:
    if args.os_extension:
        logger.info("Connecting to OpticStudio in extension mode")
        OPTICSTUDIO_BACKEND_SETTINGS["mode"] = "extension"

    if args.test:
        if args.test not in TESTS:
            logger.error("Test {} not found. Available tests: {}", args.test, list(TESTS.keys()))
            sys.exit(1)
        else:
            logger.info("Only running test {}...", args.test)
            selected_test = TESTS[args.test]
            TESTS.clear()
            TESTS[args.test] = selected_test

    backends = initialize_backends()
    eye_model = model()

    if args.save_model:
        build_and_save_model(eye_model, TEST_DATA_DIR / "eye_model", backends)

    for name, test in TESTS.items():
        for backend in backends:
            reset_backend_settings(backend)

            result_path = get_file_name(test_name=name, backend_name=str(backend.type))

            if result_path.exists() and not args.force:
                logger.info(
                    "Not generating test data for {} with backend {} already exists at {}.",
                    name,
                    backend.type,
                    result_path,
                )
                continue

            logger.info("Running {} with backend {}...", name, backend.type)
            try:
                results = test.run(model=eye_model, backend=backend)
            except Exception as e:  # noqa: BLE001
                logger.error("Error running {} with backend {}: {}", name, backend.type, e)
                continue

            results.to_csv(result_path, index=True)
            logger.success("Results saved to {}", result_path)


if __name__ == "__main__":
    filterwarnings("ignore", category=UserWarning, message="Header and row length mismatch.*")

    logger.remove()  # Remove default logger to avoid duplicate logs
    logger.add(sys.stdout, format="<level>{level} | {message}</level>", colorize=True)

    parser = argparse.ArgumentParser(description="Generate test data for analysis results tests.")
    parser.add_argument(
        "--save-model", action="store_true", help="Save the eye model used for generating the test data."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration of all test data, even if it already exists."
    )
    parser.add_argument("--test", type=str, help="Only run a specific test, identified by its name.")
    parser.add_argument(
        "--os-extension",
        action="store_true",
        help="Connect to OpticStudio in extension mode to generate the test data.",
    )

    args = parser.parse_args()
    main(args)
