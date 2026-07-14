from __future__ import annotations

import platform

import numpy as np
import pytest

from visisipy.synteyes.synteyes import (
    SyntEye,
    SyntEye3D,
    SyntEyes,
    SyntEyesBiometry,
    SyntEyesCornea,
    SyntEyesLens,
    SyntEyesMaterials,
    SyntEyesRetina,
    generate_synteyes,
)
from visisipy.wavefront import ZernikeCoefficients


@pytest.fixture
def random_state() -> np.random.Generator:
    """Fixture to set a random number generator with a fixed seed for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def sample_synteye() -> SyntEye3D:
    """Fixture for a sample SyntEyes eye model."""
    # fmt: off
    return SyntEye3D(
        biometry=SyntEyesBiometry(
            cornea_thickness=0.5675980419300763,
            anterior_chamber_depth=2.6237068010504445,
            lens_thickness=4.418377549556682,
            axial_length=24.498505342346316,
            vitreous_depth=16.68882294980911,
            pupil_diameter=5,
            retina_thickness=0.2
        ),
        cornea=SyntEyesCornea(
            anterior_zernikes=ZernikeCoefficients({1: 0.3422905828606648, 3: 0.0005918084373896628, 2: 0.0023131431725943166, 5: -0.00026351564081592986, 4: 0.20051898573989937, 6: -0.0036612315961836056, 9: -2.428312266595696e-05, 7: -2.403927691334431e-05, 8: 0.0007059334156698186, 10: 8.864882939898646e-05, 15: -3.399424117941327e-05, 13: 8.885714895108974e-05, 11: 0.002205550520562905, 12: -7.855216001463333e-05, 14: -1.570505384349936e-05, 21: -0.00011860854680563971, 19: 7.492497484619524e-05, 17: -6.127651225517059e-05, 16: -6.55477451291802e-05, 18: -3.4991858402204235e-05, 20: -1.6868186800315804e-05, 27: 2.5741322252816072e-05, 25: 4.901106964583404e-05, 23: -5.096535805866357e-05, 22: -3.9409960234172464e-05, 24: 4.943626529619928e-06, 26: -3.811789111786062e-05, 28: 1.0715046827191875e-05, 35: 1.0653441793748403e-05, 33: 4.491498417045251e-06, 31: -3.17763488862075e-06, 29: -1.3484088178570585e-05, 30: -1.443594172053776e-05, 32: -4.3937376170334406e-07, 34: -1.8896076192723605e-05, 36: 2.698192488406551e-05, 45: 2.8013263484274167e-06, 43: -1.5441741759154541e-06, 41: -7.583533435122606e-06, 39: 3.2177202266793137e-06, 37: -7.496554199440541e-06, 38: 7.089090609435584e-07, 40: 1.1694068125761121e-06, 42: -2.871193221796745e-07, 44: -4.8178166331526814e-06}),
            posterior_zernikes=ZernikeCoefficients({1: 0.4254098029065651, 3: 0.015151346008161745, 2: 0.018716241236515236, 5: -0.002894571446253507, 4: 0.250772958674681, 6: -0.011569367232794078, 9: -0.0009947693901929176, 7: 0.00022967864181157226, 8: 0.00048430334933299737, 10: 0.0002774616744770464, 15: 0.00034391399192934937, 13: -0.0005131691127042454, 11: 0.003691528043749198, 12: 0.0004634316392532736, 14: 0.0011127376473646887, 21: -0.00016148197805108445, 19: 0.00035905472032638213, 17: -0.0007099801172671901, 16: -0.0005867594542742903, 18: 0.00010592876505913449, 20: -0.0001703067033983096, 27: 0.00011447790844218325, 25: -0.0001259526428822465, 23: -6.580569413489336e-06, 22: -0.0003706205185427817, 24: -0.00012813756330976966, 26: -0.0002749176617622146, 28: -6.703428212847885e-05, 35: 1.583961893100351e-05, 33: 2.4513525025958863e-05, 31: -3.570775549361546e-05, 29: 3.95203845374714e-05, 30: 2.883462212720441e-05, 32: -2.712484730862813e-05, 34: 2.7836245784241974e-05, 36: 8.260450815834121e-05, 45: -9.890194981402716e-06, 43: -8.479011050954407e-06, 41: 4.102549733188398e-06, 39: 1.4873684209678047e-05, 37: 3.411734090933008e-05, 38: 2.4265943253665817e-05, 40: 3.189073233104356e-05, 42: 1.6333071387751156e-06, 44: -2.8446899809450242e-05}),
            anterior_norm_diameter=6.5,
            posterior_norm_diameter=6.5
        ),
        lens=SyntEyesLens(
            anterior_zernikes=ZernikeCoefficients({1: 0.0, 3: 0.0, 2: 0.0, 5: 0.0, 4: -0.003, 6: 0.0025, 9: -0.0005, 7: 0.0011, 8: 0.0009, 10: -8e-05, 15: 0.0007, 13: -0.00035, 11: -0.0023, 12: 0.0004, 14: 0.00027, 21: 0.00015, 19: 0.00025, 17: -0.00025, 16: 0.0001, 18: 0.0001, 20: 0.0001}),
            anterior_radius=9.57499968358316,
            anterior_conic=-3.1316,
            posterior_radius=-6.3395193068970705,
            posterior_conic=-1,
            anterior_norm_diameter=5.5
        ),
        materials=SyntEyesMaterials(
            cornea_index=1.376,
            aqueous_index=1.336,
            lens_index=1.4235768359806096,
            vitreous_index=1.336
        ),
        retina=SyntEyesRetina(
            radius_x=12.280656978277218,
            radius_y=11.504475212240298,
            radius_z=10.773704422659911
        )
    )
    # fmt: on


def assert_dicts_close(a: dict, b: dict, rtol: float = 1e-5, atol: float = 1e-8) -> None:
    """Assert that two dictionaries are close in value."""

    for key, value_a in a.items():
        assert key in b, f"Key {key} not found in second dictionary"

        value_b = b[key]

        if isinstance(value_a, dict) and isinstance(value_b, dict):
            assert_dicts_close(value_a, value_b, rtol=rtol, atol=atol)
        else:
            assert value_a == pytest.approx(value_b, rel=rtol, abs=atol), f"Values for key {key} are not close"


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="multivariate_normal returns different results on macOS. See https://numpy.org/neps/nep-0019-rng-policy.html#the-status-quo.",
)
def test_generate_single_synteye(sample_synteye: SyntEye3D, random_state: np.random.Generator) -> None:
    """Test generating a single SyntEyes eye model."""
    synteye = generate_synteyes(1, rng=random_state)[0]

    assert_dicts_close(synteye.to_dict(), sample_synteye.to_dict())


def test_generate_multiple_synteyes() -> None:
    """Test generating multiple SyntEyes eye models."""
    synteyes = generate_synteyes(3)

    assert len(synteyes) == 3
    assert all(isinstance(synteye, SyntEye3D) for synteye in synteyes)


@pytest.mark.parametrize("n", [0, -1])
def test_generate_invalid_number_of_synteyes(n: int) -> None:
    """Test generating an invalid number of SyntEyes eye models."""
    with pytest.raises(ValueError, match="Number of synteyes must be a positive integer"):
        generate_synteyes(n)


def test_synteye_json_roundtrip(tmp_path):
    json_file = tmp_path / "synteye.json"
    assert not json_file.exists()

    synteyes = SyntEyes(
        [
            SyntEye(biometry=synteye.biometry, cornea=synteye.cornea, lens=synteye.lens, materials=synteye.materials)
            for synteye in generate_synteyes(3)
        ]
    )

    # Save to JSON
    synteyes.save_json(json_file)
    assert json_file.exists()

    # Load from JSON
    loaded_synteyes = SyntEyes.load_json(json_file)
    assert len(loaded_synteyes) == 3
    assert loaded_synteyes == synteyes


def test_synteye3d_json_roundtrip(tmp_path):
    json_file = tmp_path / "synteye.json"
    assert not json_file.exists()

    synteyes = generate_synteyes(3)

    # Save to JSON
    synteyes.save_json(json_file)
    assert json_file.exists()

    # Load from JSON
    loaded_synteyes = SyntEyes.load_json(json_file)
    assert len(loaded_synteyes) == 3
    assert loaded_synteyes == synteyes
