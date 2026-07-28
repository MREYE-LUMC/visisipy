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
            cornea_thickness=np.float64(0.5133906811944047),
            anterior_chamber_depth=np.float64(2.769022163841836),
            lens_thickness=np.float64(4.107406492299111),
            axial_length=np.float64(24.57306753719991),
            vitreous_depth=np.float64(16.983248199864562),
            pupil_diameter=5,
            retina_thickness=0.2
        ),
        cornea=SyntEyesCornea(
            anterior_zernikes=ZernikeCoefficients({1: np.float64(0.33383641117118246), 3: np.float64(-0.00037804537100678544), 2: np.float64(0.0018277953698956585), 5: np.float64(0.00014184374570770552), 4: np.float64(0.1954617027638545), 6: np.float64(-0.004820402139643492), 9: np.float64(-5.495542568355096e-05), 7: np.float64(-0.00016681812275682142), 8: np.float64(0.0006236769613076852), 10: np.float64(4.250710281165396e-05), 15: np.float64(-3.183150857402631e-05), 13: np.float64(7.306404893589455e-05), 11: np.float64(0.0020338254477875756), 12: np.float64(-0.00011521806650768398), 14: np.float64(-3.838941306292632e-06), 21: np.float64(-0.0001425306425589151), 19: np.float64(9.62591806873507e-05), 17: np.float64(-0.00013297570409600063), 16: np.float64(-8.849111753973408e-05), 18: np.float64(-9.010001483630249e-06), 20: np.float64(-1.44043952437998e-05), 27: np.float64(1.0704417643477539e-05), 25: np.float64(6.182985065660555e-05), 23: np.float64(-7.068922027528854e-05), 22: np.float64(-7.102958325803555e-05), 24: np.float64(-2.6770891860098664e-06), 26: np.float64(-7.926467081194914e-06), 28: np.float64(-1.92081478712047e-05), 35: np.float64(1.770126374587867e-05), 33: np.float64(2.445902993651641e-05), 31: np.float64(-9.215511550395455e-06), 29: np.float64(-1.2277699978011004e-05), 30: np.float64(-2.3196931478963657e-05), 32: np.float64(2.2275659115379864e-06), 34: np.float64(-2.4863270489848628e-05), 36: np.float64(1.565374878911707e-05), 45: np.float64(1.4220542088621972e-06), 43: np.float64(5.789389384945378e-06), 41: np.float64(-1.0121992119388195e-05), 39: np.float64(4.329912992945049e-06), 37: np.float64(-7.397103513283147e-06), 38: np.float64(4.893367452889017e-06), 40: np.float64(-4.374908404418273e-06), 42: np.float64(1.3276735551724073e-06), 44: np.float64(3.5694331749876344e-06)}),
            posterior_zernikes=ZernikeCoefficients({1: np.float64(0.4146357313818231), 3: np.float64(0.0202018138273757), 2: np.float64(0.02436951223810551), 5: np.float64(-0.001843274542193113), 4: np.float64(0.24253237852351964), 6: np.float64(-0.011431926598274607), 9: np.float64(-0.0004198639594042859), 7: np.float64(-0.0013575441501551004), 8: np.float64(-0.00042643401500347356), 10: np.float64(0.0005582945560260205), 15: np.float64(0.0005636407378504364), 13: np.float64(-0.0009566976766784132), 11: np.float64(0.0021322857867674054), 12: np.float64(0.00036215694741901393), 14: np.float64(0.0009964146645045733), 21: np.float64(-0.0002597921340031438), 19: np.float64(0.0003463080046501646), 17: np.float64(-0.0009980480607076785), 16: np.float64(-0.0010562913053549586), 18: np.float64(0.00015857832934801884), 20: np.float64(-0.00010892204485177254), 27: np.float64(4.8508025382072045e-05), 25: np.float64(-0.00017156396400351645), 23: np.float64(2.9779634762349604e-05), 22: np.float64(-0.0004937896023842686), 24: np.float64(-0.00011338748866503154), 26: np.float64(-0.00022645358598634332), 28: np.float64(-6.682103146186239e-05), 35: np.float64(8.211715185379642e-05), 33: np.float64(4.596496213588015e-05), 31: np.float64(-6.077521597650732e-05), 29: np.float64(0.00011022848750666114), 30: np.float64(0.00010824129460721965), 32: np.float64(-3.717420689456883e-05), 34: np.float64(1.391711701588934e-05), 36: np.float64(0.00010067153704769786), 45: np.float64(1.905091507944274e-05), 43: np.float64(-3.4633035896069895e-06), 41: np.float64(6.967585320691337e-06), 39: np.float64(2.2279888200690227e-05), 37: np.float64(5.673597786736905e-05), 38: np.float64(2.2910513640175466e-05), 40: np.float64(2.5099028217451587e-05), 42: np.float64(5.861654693570087e-06), 44: np.float64(-3.357612244531561e-05)}),
            anterior_norm_diameter=6.5,
            posterior_norm_diameter=6.5
        ),
        lens=SyntEyesLens(
            anterior_zernikes=ZernikeCoefficients({1: np.float64(0.0), 3: np.float64(0.0), 2: np.float64(0.0), 5: np.float64(0.0), 4: np.float64(-0.003), 6: np.float64(0.0025), 9: np.float64(-0.0005), 7: np.float64(0.0011), 8: np.float64(0.0009), 10: np.float64(-8e-05), 15: np.float64(0.0007), 13: np.float64(-0.00035), 11: np.float64(-0.0023), 12: np.float64(0.0004), 14: np.float64(0.00027), 21: np.float64(0.00015), 19: np.float64(0.00025), 17: np.float64(-0.00025), 16: np.float64(0.0001), 18: np.float64(0.0001), 20: np.float64(0.0001)}),
            anterior_radius=np.float64(10.354139102546977),
            anterior_conic=-3.1316,
            posterior_radius=np.float64(-6.824054594824691),
            posterior_conic=-1,
            anterior_norm_diameter=5.5
        ),
        materials=SyntEyesMaterials(
            cornea_index=1.376,
            aqueous_index=1.336,
            lens_index=np.float64(1.431518256145379),
            vitreous_index=1.336
        ),
        retina=SyntEyesRetina(
            radius_x=np.float64(12.196827146534268),
            radius_y=np.float64(11.699968815246145),
            radius_z=np.float64(10.888088519588928)
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
    platform.machine().lower() not in {"x86_64", "amd64"},
    reason=(
        "multivariate_normal returns different results on different architectures. "
        "See https://numpy.org/neps/nep-0019-rng-policy.html#the-status-quo."
    ),
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

    for original, loaded in zip(synteyes, loaded_synteyes, strict=True):
        assert_dicts_close(original.to_dict(), loaded.to_dict())


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

    for original, loaded in zip(synteyes, loaded_synteyes, strict=True):
        assert_dicts_close(original.to_dict(), loaded.to_dict())
