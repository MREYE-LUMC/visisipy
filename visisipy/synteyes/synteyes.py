"""Generate synthetic eye models using the SyntEyes method."""

from __future__ import annotations

import importlib.resources
import json
from collections import UserList
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from scipy import stats

from visisipy.wavefront import ZernikeCoefficients

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import NDArray

__all__ = (
    "SyntEye",
    "SyntEye3D",
    "generate_synteyes",
)


@dataclass
class SyntEyesModelData:
    conv_ec_orig: NDArray[np.float64]
    avg_ec_orig: NDArray[np.float64]
    lens_za_orig: NDArray[np.float64]
    weights_orig: NDArray[np.float64]
    mu_orig: NDArray[np.float64]
    cov_orig: NDArray[np.float64]
    mu_retina_radii: NDArray[np.float64]
    cov_retina_radii: NDArray[np.float64]


@lru_cache(maxsize=1)
def load_synteyes_model_data() -> SyntEyesModelData:
    """Load the SyntEyes model data.

    Returns
    -------
    SyntEyesModelData
        A dictionary containing the SyntEyes model data.
    """
    file = importlib.resources.files(__package__) / "modeldata.npz"

    with np.load(file) as data:
        cov_orig = np.zeros((2, *data["cov_orig0"].shape))
        cov_orig[0] = nearest_psd(data["cov_orig0"])
        cov_orig[1] = data["cov_orig1"]

        return SyntEyesModelData(
            conv_ec_orig=data["conv_ec_orig"],
            avg_ec_orig=data["avg_ec_orig"],
            lens_za_orig=data["lens_za_orig"],
            weights_orig=data["weights_orig"],
            mu_orig=data["mu_orig"],
            cov_orig=cov_orig,
            mu_retina_radii=data["mu_retina_radii"],
            cov_retina_radii=data["cov_retina_radii"],
        )


def conditional_sgm(
    mu: NDArray, cov: NDArray, known_indices: list[int], known_values: NDArray | float
) -> tuple[NDArray, NDArray]:
    """Compute the conditional Gaussian distribution.

    Parameters
    ----------
    mu : np.ndarray
        Mean vector of the full Gaussian.
    cov : np.ndarray
        Covariance matrix of the full Gaussian.
    known_indices : set[int]
        Indices of variables conditioned on.
    known_values : np.ndarray | float
        Observed values for the variables in ``known_indices``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Conditional mean vector and conditional covariance matrix.
    """
    all_indices = set(range(len(mu)))
    unknown_indices = sorted(all_indices - set(known_indices))

    mu_known = mu[known_indices]
    mu_unknown = mu[unknown_indices]

    cov_known_known = cov[np.ix_(known_indices, known_indices)]
    cov_known_unknown = cov[np.ix_(known_indices, unknown_indices)]
    cov_unknown_known = cov[np.ix_(unknown_indices, known_indices)]
    cov_unknown_unknown = cov[np.ix_(unknown_indices, unknown_indices)]

    cov_known_known_inv = np.linalg.inv(cov_known_known)

    conditional_mean = mu_unknown + cov_unknown_known @ cov_known_known_inv @ (known_values - mu_known)
    conditional_cov = cov_unknown_unknown - cov_unknown_known @ cov_known_known_inv @ cov_known_unknown

    return conditional_mean, conditional_cov


@lru_cache(maxsize=10)
def zernike_index(order: int) -> list[tuple[int, int]]:
    """Create Zernike polynomial (n, m) index pairs.

    Parameters
    ----------
    order : int
        Maximum radial Zernike order.

    Returns
    -------
    np.ndarray
        Array of shape ``(k, 2)`` containing ``(n, m)`` pairs.
    """
    indices: list[tuple[int, int]] = []

    for n in range(order + 1):
        indices.extend((n, m) for m in range(-n, n + 1, 2))

    return indices


@dataclass
class SyntEye:
    """Properties of a SyntEyes eye model."""

    biometry: SyntEyesBiometry
    cornea: SyntEyesCornea
    lens: SyntEyesLens
    materials: SyntEyesMaterials

    def to_dict(self) -> dict[str, object]:
        """Convert the SyntEye instance to a dictionary.

        Returns
        -------
        dict[str, object]
            Dictionary representation of the SyntEye instance.
        """
        return asdict(self)


@dataclass
class SyntEye3D(SyntEye):
    """Properties of a 3D SyntEyes eye model.

    A 3D SyntEyes model is a SyntEye model with a retinal curvature.
    """

    retina: SyntEyesRetina

    @classmethod
    def from_synteye(cls, synteye: SyntEye, retina: SyntEyesRetina) -> SyntEye3D:
        """Create a SyntEye3D instance from a SyntEye instance and retina.

        Parameters
        ----------
        synteye : SyntEye
            The SyntEye instance to convert.
        retina : SyntEyesRetina
            The retina for the 3D eye model.

        Returns
        -------
        SyntEye3D
            A new SyntEye3D instance with the same biometry, cornea, and lens as the input SyntEye,
            but with the specified retina.
        """
        return cls(
            biometry=synteye.biometry,
            cornea=synteye.cornea,
            lens=synteye.lens,
            retina=retina,
            materials=synteye.materials,
        )


@dataclass
class SyntEyesBiometry:
    """Biometry data for a SyntEyes eye model."""

    cornea_thickness: float
    anterior_chamber_depth: float
    lens_thickness: float
    axial_length: float
    vitreous_depth: float
    pupil_diameter: float
    retina_thickness: float


@dataclass
class SyntEyesCornea:
    """Cornea data for a SyntEyes eye model."""

    anterior_zernikes: ZernikeCoefficients
    posterior_zernikes: ZernikeCoefficients


@dataclass
class SyntEyesLens:
    """Lens data for a SyntEyes eye model."""

    anterior_zernikes: ZernikeCoefficients
    anterior_radius: float
    anterior_conic: float
    posterior_radius: float
    posterior_conic: float


@dataclass
class SyntEyesRetina:
    """Retina data for a 3D SyntEyes eye model."""

    radius_x: float
    radius_y: float
    radius_z: float


@dataclass
class SyntEyesMaterials:
    """Refractive indices for a SyntEyes eye model."""

    cornea_index: float
    aqueous_index: float
    lens_index: float
    vitreous_index: float


_INDEX_ACD = 0  # Anterior chamber depth
_INDEX_LT = 1  # Lens thickness
_INDEX_AL = 2  # Axial length
_INDEX_RLA = 3  # Anterior lens radius
_INDEX_RLP = 4  # Posterior lens radius
_INDEX_CCT = 96  # Central corneal thickness
_INDEX_NUM5 = 5  # Lens shape factor
_INDICES_EIGENCORNEA = np.s_[6:18]  # Eigencornea coefficients
_INDICES_ANTERIOR_CORNEA_ZERNIKE = np.s_[6:51]  # Cornea Zernike coefficients
_INDICES_POSTERIOR_CORNEA_ZERNIKE = np.s_[51:96]  # Cornea Zernike coefficients

_SYNTEYES_RETINA_THICKNESS = 0.2  # Retina thickness in mm
_SYNTEYES_ANTERIOR_LENS_CONIC = -3.1316  # Anterior lens conic constant
_SYNTEYES_POSTERIOR_LENS_CONIC = -1  # Posterior lens con
_SYNTEYES_PUPIL_DIAMETER = 5  # Pupil diameter in mm
_SYNTEYES_CORNEA_INDEX = 1.376  # Cornea refractive index
_SYNTEYES_AQUEOUS_INDEX = 1.336  # Aqueous humor refractive index
_SYNTEYES_VITREOUS_INDEX = 1.336  # Vitreous humor refractive index


def _convert_biometry(synteyes_array: NDArray[np.float64]) -> SyntEyesBiometry:
    cornea_thickness = synteyes_array[_INDEX_CCT]
    anterior_chamber_depth = synteyes_array[_INDEX_ACD]
    lens_thickness = synteyes_array[_INDEX_LT]
    axial_length = synteyes_array[_INDEX_AL]
    vitreous_depth = (
        axial_length - anterior_chamber_depth - lens_thickness - cornea_thickness - _SYNTEYES_RETINA_THICKNESS
    )

    return SyntEyesBiometry(
        cornea_thickness=cornea_thickness,
        anterior_chamber_depth=anterior_chamber_depth,
        lens_thickness=lens_thickness,
        axial_length=axial_length,
        vitreous_depth=vitreous_depth,
        pupil_diameter=_SYNTEYES_PUPIL_DIAMETER,
        retina_thickness=_SYNTEYES_RETINA_THICKNESS,
    )


def _convert_cornea(synteyes_array: NDArray[np.float64]) -> SyntEyesCornea:
    zernike_indices = zernike_index(8)

    anterior_zernike_terms = synteyes_array[_INDICES_ANTERIOR_CORNEA_ZERNIKE]
    anterior_zernikes = ZernikeCoefficients({
        (n, m): anterior_zernike_terms[i] for i, (n, m) in enumerate(zernike_indices)
    })

    posterior_zernike_terms = synteyes_array[_INDICES_POSTERIOR_CORNEA_ZERNIKE]
    posterior_zernikes = ZernikeCoefficients({
        (n, m): posterior_zernike_terms[i] for i, (n, m) in enumerate(zernike_indices)
    })

    return SyntEyesCornea(
        anterior_zernikes=anterior_zernikes,
        posterior_zernikes=posterior_zernikes,
    )


def _convert_lens(synteyes_array: NDArray[np.float64], lens_zernikes: NDArray[np.float64]) -> SyntEyesLens:
    zernike_indices = zernike_index(6)
    anterior_zernikes = ZernikeCoefficients({
        (n, m): coefficient for coefficient, (n, m) in zip(lens_zernikes, zernike_indices, strict=False)
    })

    return SyntEyesLens(
        anterior_zernikes=anterior_zernikes,
        anterior_radius=synteyes_array[_INDEX_RLA],
        anterior_conic=_SYNTEYES_ANTERIOR_LENS_CONIC,
        posterior_radius=synteyes_array[_INDEX_RLP],
        posterior_conic=_SYNTEYES_POSTERIOR_LENS_CONIC,
    )


def _convert_materials(
    synteyes_array: NDArray[np.float64], biometry: SyntEyesBiometry, lens: SyntEyesLens
) -> SyntEyesMaterials:
    num_5 = synteyes_array[_INDEX_NUM5]

    lens_index = (
        1000
        * (
            _SYNTEYES_VITREOUS_INDEX * (biometry.lens_thickness - lens.anterior_radius)
            + _SYNTEYES_AQUEOUS_INDEX * (biometry.lens_thickness + lens.posterior_radius)
        )
        + num_5 * lens.anterior_radius * lens.posterior_radius
        - np.sqrt(
            -4
            * 1e6
            * _SYNTEYES_AQUEOUS_INDEX
            * _SYNTEYES_VITREOUS_INDEX
            * biometry.lens_thickness
            * (biometry.lens_thickness - lens.anterior_radius + lens.posterior_radius)
            + (
                1000 * _SYNTEYES_VITREOUS_INDEX * (-1 * biometry.lens_thickness + lens.anterior_radius)
                + 1000 * _SYNTEYES_AQUEOUS_INDEX * (-1 * biometry.lens_thickness - 1 * lens.posterior_radius)
                - num_5 * lens.anterior_radius * lens.posterior_radius
            )
            ** 2
        )
    ) / (2000 * (biometry.lens_thickness - lens.anterior_radius + lens.posterior_radius))

    return SyntEyesMaterials(
        cornea_index=_SYNTEYES_CORNEA_INDEX,
        aqueous_index=_SYNTEYES_AQUEOUS_INDEX,
        lens_index=lens_index,
        vitreous_index=_SYNTEYES_VITREOUS_INDEX,
    )


def convert_to_single_orig_synteyes(
    eigencornea: NDArray, conv_ec: NDArray, avg_ec: NDArray, lens_za: NDArray
) -> SyntEye:
    """Convert one eigencornea sample to SyntEyes format.

    Parameters
    ----------
    eigencornea : np.ndarray
        One eigencornea sample containing biometric latent variables.
    conv_ec : np.ndarray
        Conversion matrix from eigencornea coefficients to corneal terms.
    avg_ec : np.ndarray
        Mean corneal terms added after conversion.
    lens_za : np.ndarray
        Lens Zernike coefficients.

    Returns
    -------
    pandas.DataFrame
        Single-row dataframe with SyntEyes fields.
    """
    zernikes = 0.001 * eigencornea[np.newaxis, _INDICES_EIGENCORNEA] @ conv_ec.T
    zernikes += avg_ec  # avg_ec has shape (1, 91)

    synteyes_array = np.append(eigencornea[:6], zernikes)

    biometry = _convert_biometry(synteyes_array)
    cornea = _convert_cornea(synteyes_array)
    lens = _convert_lens(synteyes_array, lens_za)
    materials = _convert_materials(synteyes_array, biometry, lens)

    return SyntEye(
        biometry=biometry,
        cornea=cornea,
        lens=lens,
        materials=materials,
    )


def sample_retina_curvature(
    axial_lengths: NDArray[np.float64] | list[float],
    retina_thicknesses: NDArray[np.float64] | list[float],
    mu_retina: NDArray,
    cov_retina: NDArray,
) -> list[SyntEyesRetina]:
    """Sample retinal curvature values conditioned on axial length.

    Parameters
    ----------
    axial_lengths : NDArray[np.float64] | list[float]
        List of axial lengths for which to calculate retinal curvature, including the retina thickness.
    retina_thicknesses : NDArray[np.float64] | list[float]
        List of retina thicknesses corresponding to each axial length.
    mu_retina : np.ndarray
        Mean vector for the retina model.
    cov_retina : np.ndarray
        Covariance matrix for the retina model.

    Returns
    -------
    list[SyntEyesRetina]
        List of retinal curvature values.
    """
    axial_lengths = np.array(axial_lengths) - np.array(retina_thicknesses)

    cond_sgm = np.zeros((len(axial_lengths), 3))
    for i, axial_length in enumerate(axial_lengths):
        mean, covariance = conditional_sgm(mu_retina, cov_retina, [0], axial_length)
        cond_sgm[i] = stats.multivariate_normal.rvs(mean=mean, cov=covariance, size=1)

    return [SyntEyesRetina(radius_x=rx, radius_y=ry, radius_z=rz) for rx, ry, rz in cond_sgm]


def create_mgmm_data(
    mu_c0: NDArray,
    mu_c1: NDArray,
    cov_c0: NDArray,
    cov_c1: NDArray,
    w_c0: float,
    w_c1: float,
    n: int,
) -> NDArray:
    """Sample from a weighted two-component Gaussian mixture model.

    Parameters
    ----------
    mu_c0 : np.ndarray
        Mean vector of component 0.
    mu_c1 : np.ndarray
        Mean vector of component 1.
    cov_c0 : np.ndarray
        Covariance matrix of component 0.
    cov_c1 : np.ndarray
        Covariance matrix of component 1.
    w_c0 : float
        Weight for component 0 sample.
    w_c1 : float
        Weight for component 1 sample.
    n : int
        Number of samples to generate.

    Returns
    -------
    np.ndarray
        Generated samples in latent eigencornea space.
    """
    comp0 = stats.multivariate_normal.rvs(mu_c0, cov_c0, size=n)
    comp1 = stats.multivariate_normal.rvs(mu_c1, cov_c1, size=n)
    return w_c0 * comp0 + w_c1 * comp1


def nearest_psd(matrix: NDArray) -> NDArray:
    """Project a matrix to a positive semi-definite approximation.

    Parameters
    ----------
    matrix : np.ndarray
        Input square matrix.

    Returns
    -------
    np.ndarray
        Matrix with eigenvalues clipped to a small positive threshold.
    """
    eigval, eigvec = np.linalg.eig(matrix)
    return eigvec @ np.diag(np.maximum(eigval, 1e-6)) @ eigvec.T


S = TypeVar("S", SyntEye, SyntEye3D, SyntEye | SyntEye3D)


class SyntEyes(UserList[S]):
    """A list of SyntEye or SyntEye3D instances."""

    def __init__(self, initlist: list[S] | None = None) -> None:
        """Initialize the SyntEyes list.

        Parameters
        ----------
        initlist : list[S] | None, optional
            Initial list of SyntEye or SyntEye3D instances. If None, an empty list is created.
        """
        super().__init__(initlist if initlist is not None else [])

    def save_json(self, filename: PathLike | str) -> None:
        """Save the SyntEyes list to a JSON file.

        Parameters
        ----------
        filename : PathLike | str
            The path to the JSON file where the data will be saved.

        Raises
        ------
        FileNotFoundError
            If the directory of the specified filename does not exist.
        """
        filename = Path(filename).resolve()

        if not filename.parent.exists():
            raise FileNotFoundError(f"Directory does not exist: {filename.parent}")

        filename.write_text(json.dumps([eye.to_dict() for eye in self], indent=4))

    @classmethod
    def load_json(cls, filename: PathLike | str) -> SyntEyes:
        """Load SyntEyes data from a JSON file.

        Parameters
        ----------
        filename : PathLike | str
            The path to the JSON file from which to load the data.

        Returns
        -------
        SyntEyes
            A SyntEyes instance containing the loaded data.

        Raises
        ------
        FileNotFoundError
            If the specified JSON file does not exist.
        """
        filename = Path(filename).resolve()

        if not filename.exists():
            raise FileNotFoundError(f"File does not exist: {filename}")

        data = json.loads(filename.read_text(encoding="utf-8"))

        result: SyntEyes[SyntEye3D | SyntEye] = cls()

        for item in data:
            item["cornea"]["anterior_zernikes"] = ZernikeCoefficients({
                int(k): v for k, v in item["cornea"]["anterior_zernikes"].items()
            })
            item["cornea"]["posterior_zernikes"] = ZernikeCoefficients({
                int(k): v for k, v in item["cornea"]["posterior_zernikes"].items()
            })
            item["lens"]["anterior_zernikes"] = ZernikeCoefficients({
                int(k): v for k, v in item["lens"]["anterior_zernikes"].items()
            })

            if "retina" in item:
                result.append(SyntEye3D(**item))
            else:
                result.append(SyntEye(**item))

        return result


def generate_synteyes(n: int) -> SyntEyes[SyntEye3D]:
    """Generate `n` 3D SyntEyes eye models.

    Parameters
    ----------
    n : int
        Number of 3D SyntEyes eye models to generate.

    Returns
    -------
    SyntEyes
        List of generated 3D SyntEyes eye models.

    Raises
    ------
    ValueError
        If `n` is not a positive integer.
    """
    if n <= 0:
        raise ValueError("Number of samples must be a positive integer.")

    model_data = load_synteyes_model_data()

    eigencorneas = create_mgmm_data(
        model_data.mu_orig[0],
        model_data.mu_orig[1],
        model_data.cov_orig[0],
        model_data.cov_orig[1],
        model_data.weights_orig[0],
        model_data.weights_orig[1],
        n,
    ).reshape(n, -1)

    axial_lengths = eigencorneas[:, _INDEX_AL]
    retina_thicknesses = np.full_like(axial_lengths, _SYNTEYES_RETINA_THICKNESS)

    retinas = sample_retina_curvature(
        axial_lengths, retina_thicknesses, model_data.mu_retina_radii, model_data.cov_retina_radii
    )

    synteyes_3d = SyntEyes[SyntEye3D]()

    for eigencornea, retina in zip(eigencorneas, retinas, strict=True):
        synteye = convert_to_single_orig_synteyes(
            eigencornea, conv_ec=model_data.conv_ec_orig, avg_ec=model_data.avg_ec_orig, lens_za=model_data.lens_za_orig
        )
        synteye_3d = SyntEye3D.from_synteye(synteye, retina)
        synteyes_3d.append(synteye_3d)

    return synteyes_3d
