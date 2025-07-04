"""Optical properties of materials used in eye models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

__all__ = (
    "EyeMaterials",
    "MaterialModel",
    "NavarroMaterials",
    "NavarroMaterials458",
    "NavarroMaterials543",
    "NavarroMaterials589",
    "NavarroMaterials633",
)


@dataclass
class MaterialModel:
    """Optical properties of a material.

    Attributes
    ----------
    refractive_index : float
        The refractive index of the material.
    abbe_number : float
        The Abbe number (or V number) of the material, which is a measure of the material's dispersion
        (variation of refractive index with wavelength).
    partial_dispersion : float
        The partial dispersion of the material, which is the difference in the refractive index of the material
        at two specific wavelengths.
    """

    refractive_index: float
    abbe_number: float
    partial_dispersion: float


@dataclass
class EyeMaterials:
    """Material parameters of the elements of an eye.

    Attributes
    ----------
    cornea : MaterialModel
        Refractive model of the cornea.
    aqueous : MaterialModel
        Refractive model of the aqueous humour.
    lens : MaterialModel
        Refractive model of the crystalline lens.
    vitreous : MaterialModel
        Refractive model of the vitreous humour.
    """

    cornea: MaterialModel | str
    aqueous: MaterialModel | str
    lens: MaterialModel | str
    vitreous: MaterialModel | str


def _material_model_factory(refractive_index, abbe_number, partial_dispersion) -> Callable[[], MaterialModel]:
    def factory() -> MaterialModel:
        return MaterialModel(refractive_index, abbe_number, partial_dispersion)

    return factory


@dataclass
class NavarroMaterials(EyeMaterials):
    """Material parameters of an eye, according to the Navarro model [1]_.

    The Navarro model defines refractive indices at various wavelengths.
    These data were used to fit the material models in OpticStudio.

    Attributes
    ----------
    cornea : MaterialModel
        Refractive model of the cornea.
    aqueous : MaterialModel
        Refractive model of the aqueous humour.
    lens : MaterialModel
        Refractive model of the crystalline lens.
    vitreous : MaterialModel
        Refractive model of the vitreous humour.

    References
    ----------
    .. [1] Escudero-Sanz, I., & Navarro, R. (1999).
       Off-axis aberrations of a wide-angle schematic eye model.
       JOSA A, 16(8), 1881-1891. https://doi.org/10.1364/JOSAA.16.001881
    """

    cornea: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.3760893927086157,
            abbe_number=53.049630674367904,
            partial_dispersion=0,
        )
    )
    aqueous: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.337454140357322,
            abbe_number=48.49905307541748,
            partial_dispersion=0,
        )
    )
    lens: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.4201120875715691,
            abbe_number=46.3823857404518,
            partial_dispersion=0,
        )
    )
    vitreous: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.3360870349145728,
            abbe_number=49.685360416317124,
            partial_dispersion=0,
        )
    )


@dataclass
class NavarroMaterials458(EyeMaterials):
    """Material parameters of an eye for a wavelength of 458 nm, according to the Navarro model [1]_.

    Attributes
    ----------
    cornea : MaterialModel
        Refractive model of the cornea.
    aqueous : MaterialModel
        Refractive model of the aqueous humour.
    lens : MaterialModel
        Refractive model of the crystalline lens.
    vitreous : MaterialModel
        Refractive model of the vitreous humour.

    References
    ----------
    .. [1] Escudero-Sanz, I., & Navarro, R. (1999).
       Off-axis aberrations of a wide-angle schematic eye model.
       JOSA A, 16(8), 1881-1891. https://doi.org/10.1364/JOSAA.16.001881
    """

    cornea: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.3828,
            abbe_number=0,
            partial_dispersion=0,
        )
    )
    aqueous: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.3445,
            abbe_number=0,
            partial_dispersion=0,
        )
    )
    lens: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.4292,
            abbe_number=0,
            partial_dispersion=0,
        )
    )
    vitreous: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.3428,
            abbe_number=0,
            partial_dispersion=0,
        )
    )


@dataclass
class NavarroMaterials543(EyeMaterials):
    """Material parameters of an eye for a wavelength of 543 nm, according to the Navarro model [1]_.

    Attributes
    ----------
    cornea : MaterialModel
        Refractive model of the cornea.
    aqueous : MaterialModel
        Refractive model of the aqueous humour.
    lens : MaterialModel
        Refractive model of the crystalline lens.
    vitreous : MaterialModel
        Refractive model of the vitreous humour.

    References
    ----------
    .. [1] Escudero-Sanz, I., & Navarro, R. (1999).
       Off-axis aberrations of a wide-angle schematic eye model.
       JOSA A, 16(8), 1881-1891. https://doi.org/10.1364/JOSAA.16.001881
    """

    cornea: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.3777,
            abbe_number=0,
            partial_dispersion=0,
        )
    )
    aqueous: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.3391,
            abbe_number=0,
            partial_dispersion=0,
        )
    )
    lens: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.4222,
            abbe_number=0,
            partial_dispersion=0,
        )
    )
    vitreous: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.3377,
            abbe_number=0,
            partial_dispersion=0,
        )
    )


@dataclass
class NavarroMaterials589(EyeMaterials):
    """Material parameters of an eye for a wavelength of 589.3 nm, according to the Navarro model [1]_.

    Attributes
    ----------
    cornea : MaterialModel
        Refractive model of the cornea.
    aqueous : MaterialModel
        Refractive model of the aqueous humour.
    lens : MaterialModel
        Refractive model of the crystalline lens.
    vitreous : MaterialModel
        Refractive model of the vitreous humour.

    References
    ----------
    .. [1] Escudero-Sanz, I., & Navarro, R. (1999).
       Off-axis aberrations of a wide-angle schematic eye model.
       JOSA A, 16(8), 1881-1891. https://doi.org/10.1364/JOSAA.16.001881
    """

    cornea: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.376,
            abbe_number=0,
            partial_dispersion=0,
        )
    )
    aqueous: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.3374,
            abbe_number=0,
            partial_dispersion=0,
        )
    )
    lens: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.42,
            abbe_number=0,
            partial_dispersion=0,
        )
    )
    vitreous: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.336,
            abbe_number=0,
            partial_dispersion=0,
        )
    )


@dataclass
class NavarroMaterials633(EyeMaterials):
    """Material parameters of an eye for a wavelength of 632.8 nm, according to the Navarro model [1]_.

    Attributes
    ----------
    cornea : MaterialModel
        Refractive model of the cornea.
    aqueous : MaterialModel
        Refractive model of the aqueous humour.
    lens : MaterialModel
        Refractive model of the crystalline lens.
    vitreous : MaterialModel
        Refractive model of the vitreous humour.

    References
    ----------
    .. [1] Escudero-Sanz, I., & Navarro, R. (1999).
       Off-axis aberrations of a wide-angle schematic eye model.
       JOSA A, 16(8), 1881-1891. https://doi.org/10.1364/JOSAA.16.001881
    """

    cornea: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.3747,
            abbe_number=0,
            partial_dispersion=0,
        )
    )
    aqueous: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.336,
            abbe_number=0,
            partial_dispersion=0,
        )
    )
    lens: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.4183,
            abbe_number=0,
            partial_dispersion=0,
        )
    )
    vitreous: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.3347,
            abbe_number=0,
            partial_dispersion=0,
        )
    )
