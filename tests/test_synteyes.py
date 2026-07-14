from __future__ import annotations

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


@pytest.fixture(scope="session")
def sample_synteye() -> SyntEye3D:
    """Fixture for a sample SyntEyes eye model."""
    # fmt: off
    return SyntEye3D(
        biometry=SyntEyesBiometry(
            cornea_thickness=0.5427433554165672,
            anterior_chamber_depth=2.5286723327544287,
            lens_thickness=3.9340384335606515,
            axial_length=21.974074563672808,
            vitreous_depth=14.768620441941161,
            pupil_diameter=5,
            retina_thickness=0.2
        ),
        cornea=SyntEyesCornea(
            anterior_zernikes=ZernikeCoefficients({1: 0.3512463515579083, 3: -0.0018489019015875653, 2: -0.0006129138268568134, 5: 0.0003042334438621534, 4: 0.20547893828179403, 6: -0.0026312458125275244, 9: -5.470947239291036e-05, 7: -0.00041684456389038695, 8: -0.00011316402390960223, 10: 3.2324415289863653e-06, 15: -1.1828865775240329e-05, 13: 2.7741058902826434e-05, 11: 0.002077132991355179, 12: -5.790910175605664e-05, 14: -8.36319438998638e-06, 21: -5.146149557126834e-05, 19: 2.4795220292956334e-05, 17: -5.430907208759868e-05, 16: -4.837858657087832e-05, 18: 7.651043777462438e-06, 20: 7.968785627619223e-06, 27: -1.5357342108753956e-06, 25: 4.653732251504018e-05, 23: -4.5186883301294796e-05, 22: -1.0128725045770642e-05, 24: 1.4357971213157333e-06, 26: -2.056257960555457e-05, 28: -9.843996798764055e-06, 35: 6.547887698160956e-07, 33: 1.1508531175286087e-05, 31: 4.975958878163364e-06, 29: -1.4683548954738188e-05, 30: -9.030901992957e-06, 32: 1.4118787376454826e-06, 34: -1.2425815584257951e-05, 36: 1.8232017571158132e-06, 45: 3.093899631169862e-06, 43: 5.422062517734378e-06, 41: -8.602715128646516e-06, 39: 6.479520450484951e-06, 37: -6.593674873856781e-06, 38: 5.193661956986361e-06, 40: 1.8559446394837385e-06, 42: -2.1784891768416175e-06, 44: 2.82006416299337e-06}),
            posterior_zernikes=ZernikeCoefficients({1: 0.4398497818509853, 3: 0.021855672329063984, 2: 0.017775896367821686, 5: 0.00024231703094581268, 4: 0.26007146052082125, 6: -0.007783166906094969, 9: 0.0004723899411912592, 7: -8.588991658436038e-05, 8: 8.571802417903544e-05, 10: 0.00012589781032179087, 15: 0.0005464375749646923, 13: -0.0006719412705913741, 11: 0.004218357420509283, 12: 0.00014677447766506038, 14: 0.0005657443161084066, 21: 8.487647716452732e-05, 19: 6.148273696506739e-05, 17: -0.0008529127447727759, 16: -0.0005280374029245139, 18: 0.0001653287880747565, 20: 0.0002430483967664433, 27: -0.00015110668086398303, 25: -9.790832284425e-05, 23: -0.0001325741769448228, 22: -0.000573766016046664, 24: -7.699038053639934e-05, 26: -0.0002795733968032901, 28: -6.236956845809394e-05, 35: 6.333992018877965e-05, 33: 5.621516343853034e-06, 31: -1.3116471398298991e-05, 29: 2.9093329786887785e-05, 30: -2.4658657401438137e-06, 32: -3.792373743321204e-05, 34: -6.672574638814117e-06, 36: 3.124776095246332e-05, 45: 2.2334049257727898e-05, 43: 1.0814128348665675e-05, 41: -7.4179909855514995e-06, 39: 4.4806296937629966e-05, 37: 7.493576575251603e-05, 38: 2.4123173745206217e-05, 40: 3.808539365378676e-05, 42: 2.8390892161894676e-06, 44: -2.2133973089037835e-05}),
            anterior_norm_diameter=6.5,
            posterior_norm_diameter=6.5
        ),
        lens=SyntEyesLens(
            anterior_zernikes=ZernikeCoefficients({1: 0.0, 3: 0.0, 2: 0.0, 5: 0.0, 4: -0.003, 6: 0.0025, 9: -0.0005, 7: 0.0011, 8: 0.0009, 10: -8e-05, 15: 0.0007, 13: -0.00035, 11: -0.0023, 12: 0.0004, 14: 0.00027, 21: 0.00015, 19: 0.00025, 17: -0.00025, 16: 0.0001, 18: 0.0001, 20: 0.0001}),
            anterior_radius=10.361422659208078,
            anterior_conic=-3.1316,
            posterior_radius=-6.827593603714798,
            posterior_conic=-1,
            anterior_norm_diameter=5.5
        ),
        materials=SyntEyesMaterials(
            cornea_index=1.376,
            aqueous_index=1.336,
            lens_index=1.441546369326015,
            vitreous_index=1.336
        ),
        retina=SyntEyesRetina(
            radius_x=11.558551033515554,
            radius_y=10.49646324639398,
            radius_z=9.66554610405957
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


def test_generate_single_synteye(sample_synteye: SyntEye3D) -> None:
    """Test generating a single SyntEyes eye model."""
    np.random.seed(42)
    synteye = generate_synteyes(1)[0]

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
