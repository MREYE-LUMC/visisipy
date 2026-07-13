"""Escudero-Sanz and Navarro wide-angle schematic eye.

This module does not provide an eye model class, because the default eye model in visisipy
is already based on the Navarro geometry and materials.
"""

from __future__ import annotations

from visisipy.models.base import EyeModel
from visisipy.models.geometry import (
    BiconicSurface,
    EyeGeometry,
    StandardSurface,
    Stop,
    ZernikeStandardSagSurface,
)
from visisipy.models.materials import EyeMaterials, MaterialModel
from visisipy.synteyes.synteyes import SyntEye3D, generate_synteyes

__all__ = ("SyntEyesGeometry",)


class SyntEyesGeometry(
    EyeGeometry[
        ZernikeStandardSagSurface,
        ZernikeStandardSagSurface,
        Stop,
        ZernikeStandardSagSurface,
        StandardSurface,
        BiconicSurface,
    ]
):
    """Geometric parameters of an eye model generated with SyntEyes.

    This schematic eye is based on the Navarro model as described in [1]_.
    Sizes are specified in mm.

    Attributes
    ----------
    cornea_front : ZernikeStandardSagSurface
        The front surface of the cornea.
    cornea_back : ZernikeStandardSagSurface
        The back surface of the cornea.
    pupil : Stop
        The pupil of the eye.
    lens_front : ZernikeStandardSagSurface
        The front surface of the lens.
    lens_back : StandardSurface
        The back surface of the lens.
    retina : StandardSurface
        The retina of the eye.

    References
    ----------
    .. [1] Escudero-Sanz, I., & Navarro, R. (1999). Off-axis aberrations of a wide-angle schematic eye model.
        JOSA A, 16(8), 1881-1891. https://doi.org/10.1364/JOSAA.16.001881

    Examples
    --------
    Use the default Navarro geometry:

    >>> from visisipy import NavarroGeometry
    >>> geometry = NavarroGeometry()

    Create a Navarro geometry with a custom retina:

    >>> geometry = NavarroGeometry(
    ...     retina=StandardSurface(radius=-12.5, asphericity=0.5)
    ... )

    Create a default Navarro geometry and change only the lens back radius:

    >>> geometry = NavarroGeometry()
    >>> geometry.lens_back.radius = -5.8
    """

    def __init__(self, synteye: SyntEye3D) -> None:
        self.cornea_front = ZernikeStandardSagSurface(
            radius=float("inf"),
            asphericity=0,
            thickness=synteye.biometry.cornea_thickness,
            zernike_coefficients=synteye.cornea.anterior_zernikes,
            norm_radius=synteye.cornea.anterior_norm_diameter / 2,
        )
        self.cornea_back = ZernikeStandardSagSurface(
            radius=float("inf"),
            asphericity=0,
            thickness=synteye.biometry.anterior_chamber_depth,
            zernike_coefficients=synteye.cornea.posterior_zernikes,
            norm_radius=synteye.cornea.posterior_norm_diameter / 2,
        )
        self.pupil = Stop(semi_diameter=synteye.biometry.pupil_diameter / 2)
        self.lens_front = ZernikeStandardSagSurface(
            radius=synteye.lens.anterior_radius,
            asphericity=synteye.lens.anterior_conic,
            thickness=synteye.biometry.lens_thickness,
            zernike_coefficients=synteye.lens.anterior_zernikes,
            norm_radius=synteye.lens.anterior_norm_diameter / 2,
        )
        self.lens_back = StandardSurface(
            radius=synteye.lens.posterior_radius,
            asphericity=synteye.lens.posterior_conic,
            thickness=synteye.biometry.vitreous_depth,
        )

        retina_radius_y = synteye.retina.radius_y**2 / synteye.retina.radius_z
        retina_asphericity_y = (synteye.retina.radius_y / synteye.retina.radius_z) ** 2 - 1
        retina_radius_x = synteye.retina.radius_x**2 / synteye.retina.radius_z
        retina_asphericity_x = (synteye.retina.radius_x / synteye.retina.radius_z) ** 2 - 1

        self.retina = BiconicSurface(
            radius=retina_radius_y,
            radius_x=retina_radius_x,
            asphericity=retina_asphericity_y,
            asphericity_x=retina_asphericity_x,
        )


class SyntEyesEyeModel(EyeModel):
    """SyntEyes randomly generated schematic eye model.

    See Also
    --------
    SyntEyesGeometry : Geometric parameters of an eye model generated with SyntEyes.
    visisipy.models.materials.SyntEyesMaterials : Optical materials of an eye model generated with SyntEyes.
    """

    def __init__(self, synteye: SyntEye3D | None = None) -> None:
        """Create a SyntEyes schematic eye model.

        If a SyntEye3D object is not provided, a random SyntEyes eye model will be generated.

        Parameters
        ----------
        synteye : SyntEye3D | None
            A SyntEye3D object containing the geometric and optical properties of the eye model.
        """
        if synteye is None:
            synteye = generate_synteyes(1)[0]

        geometry = SyntEyesGeometry(synteye)
        materials = EyeMaterials(
            cornea=MaterialModel(refractive_index=synteye.materials.cornea_index),
            aqueous=MaterialModel(refractive_index=synteye.materials.aqueous_index),
            lens=MaterialModel(refractive_index=synteye.materials.lens_index),
            vitreous=MaterialModel(refractive_index=synteye.materials.vitreous_index),
        )
        super().__init__(geometry=geometry, materials=materials)

    @classmethod
    def generate(cls, n: int) -> SyntEyesEyeModel | list[SyntEyesEyeModel]:
        """Generate one or multiple SyntEyes schematic eye models.

        Parameters
        ----------
        n : int
            The number of SyntEyes eye models to generate.

        Returns
        -------
        SyntEyesEyeModel | list[SyntEyesEyeModel]
            If n is 1, returns a single SyntEyesEyeModel. Otherwise, returns a list of SyntEyesEyeModel objects.

        Raises
        ------
        ValueError
            If n is not a positive integer.
        """
        if n < 1:
            raise ValueError("n must be a positive integer.")

        synteyes = generate_synteyes(n)

        if n == 1:
            return cls(synteyes[0])

        return [cls(synteye) for synteye in synteyes]
