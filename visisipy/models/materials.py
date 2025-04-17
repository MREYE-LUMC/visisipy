from dataclasses import dataclass, field

__all__ = ("MaterialModel", "EyeMaterials", "NavarroMaterials")

from typing import Callable


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

    cornea: MaterialModel
    aqueous: MaterialModel
    lens: MaterialModel
    vitreous: MaterialModel


def _material_model_factory(refractive_index, abbe_number, partial_dispersion) -> Callable[[], MaterialModel]:
    def factory() -> MaterialModel:
        return MaterialModel(refractive_index, abbe_number, partial_dispersion)

    return factory


@dataclass
class NavarroMaterials(EyeMaterials):
    """Material parameters of an eye, according to the Navarro model.

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
    """

    cornea: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.37602751686,
            abbe_number=56.9362270454,
            partial_dispersion=0.0633737882164,
        )
    )
    aqueous: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.33738703254,
            abbe_number=49.0704608205,
            partial_dispersion=0.0618839248407,
        )
    )
    lens: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.4200127433,
            abbe_number=48.0785554825,
            partial_dispersion=0.0838140446604,
        )
    )
    vitreous: MaterialModel = field(
        default_factory=_material_model_factory(
            refractive_index=1.33602751687,
            abbe_number=50.8796247462,
            partial_dispersion=0.0531865765832,
        )
    )
