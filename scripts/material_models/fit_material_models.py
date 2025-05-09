from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import zospy as zp
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.lines import Line2D
from optiland.materials import AbbeMaterial
from scipy.optimize import curve_fit

from visisipy.models.materials import (
    NavarroMaterials458,
    NavarroMaterials543,
    NavarroMaterials589,
    NavarroMaterials633,
)

if TYPE_CHECKING:
    from zospy.zpcore import OpticStudioSystem


NAVARRO_MATERIAL_MODELS = {
    0.458: NavarroMaterials458(),
    0.543: NavarroMaterials543(),
    0.5893: NavarroMaterials589(),
    0.6328: NavarroMaterials633(),
}
STRUCTURES = ("cornea", "aqueous", "lens", "vitreous")


def create_model(oss: OpticStudioSystem, *, fit_dispersion: bool = False) -> None:
    oss.new()
    oss.make_sequential()

    oss.SystemData.Fields.SetFieldType(zp.constants.SystemData.FieldType.Angle)
    oss.SystemData.Fields.GetField(1).X = 0
    oss.SystemData.Fields.GetField(1).Y = 0

    assert oss.SystemData.Fields.NumberOfFields == 1, "Number of fields should be 1"

    for i, w in enumerate(NAVARRO_MATERIAL_MODELS.keys()):
        if i == 0:
            oss.SystemData.Wavelengths.GetWavelength(1).Wavelength = w
        else:
            oss.SystemData.Wavelengths.AddWavelength(w, 1)

    assert oss.SystemData.Wavelengths.NumberOfWavelengths == len(NAVARRO_MATERIAL_MODELS), (
        f"Number of wavelengths should be {len(NAVARRO_MATERIAL_MODELS)}"
    )

    oss.SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.FloatByStopSize

    aperture = oss.LDE.GetSurfaceAt(1)
    aperture.SemiDiameter = 5

    lens_front = oss.LDE.InsertNewSurfaceAt(2)
    lens_front.Thickness = 10
    lens_front.Radius = 10
    material_model = zp.solvers.material_model(
        lens_front.MaterialCell, refractive_index=1, abbe_number=0, partial_dispersion=0
    )
    material_model.VaryIndex = True
    material_model.VaryAbbe = True
    material_model.VarydPgF = fit_dispersion

    lens_front.MaterialCell.SetSolveData(material_model)

    lens_back = oss.LDE.InsertNewSurfaceAt(3)
    lens_back.Thickness = 0
    lens_back.Radius = -10


def _get_wavelength_number(oss: OpticStudioSystem, wavelength: float) -> int:
    for i in range(1, oss.SystemData.Wavelengths.NumberOfWavelengths + 1):
        if oss.SystemData.Wavelengths.GetWavelength(i).Wavelength == wavelength:
            return i

    raise ValueError(f"Wavelength {wavelength} not found in system.")


def build_merit_function(oss: OpticStudioSystem, refractive_indices: dict[float, float]) -> None:
    oss.MFE.ShowEditor()
    oss.MFE.DeleteAllRows()

    comment = oss.MFE.GetOperandAt(1)
    comment.ChangeType(zp.constants.Editors.MFE.MeritOperandType.BLNK)
    comment.GetCellAt(1).Value = "Refractive indices per wavelength"

    for wavelength, refractive_index in refractive_indices.items():
        operand = oss.MFE.InsertNewOperandAt(oss.MFE.NumberOfOperands + 1)
        operand.ChangeType(zp.constants.Editors.MFE.MeritOperandType.INDX)
        operand.GetCellAt(2).Value = str(2)  # Surface number
        operand.GetCellAt(3).Value = str(_get_wavelength_number(oss, wavelength))  # Wavelength number
        operand.Target = refractive_index
        operand.Weight = 1.0


class OptimizationResult(NamedTuple):
    target: float
    value: float

    @property
    def difference(self) -> float:
        return self.value - self.target


class MaterialModel(NamedTuple):
    refractive_index: float
    abbe_number: float = 0
    partial_dispersion: float = 0


def run_optimization(
    oss: OpticStudioSystem,
) -> tuple[MaterialModel, dict[float, OptimizationResult]]:
    if oss.Tools.CurrentTool is not None:
        oss.Tools.CurrentTool.Close()

    local_optimization = oss.Tools.OpenLocalOptimization()
    local_optimization.Algorithm = zp.constants.Tools.Optimization.OptimizationAlgorithm.DampedLeastSquares
    local_optimization.NumberOfCores = 20
    local_optimization.Cycles = zp.constants.Tools.Optimization.OptimizationCycles.Automatic

    local_optimization.RunAndWaitForCompletion()

    if not local_optimization.Succeeded:
        print("Optimization failed", local_optimization.ErrorMessage)
        sys.exit(2)

    local_optimization.Close()

    # oss.MFE.CalculateMeritFunction()

    result = {}

    for i in range(1, oss.MFE.NumberOfOperands + 1):
        operand = oss.MFE.GetOperandAt(i)
        if operand.TypeName == "INDX":
            wavelength = oss.SystemData.Wavelengths.GetWavelength(operand.GetCellAt(3).IntegerValue).Wavelength
            target = operand.Target
            value = operand.Value

            result[wavelength] = OptimizationResult(target, value)

    solve_data = oss.LDE.GetSurfaceAt(2).MaterialCell.GetSolveData()

    return MaterialModel(
        solve_data._S_MaterialModel.IndexNd,  # noqa: SLF001
        solve_data._S_MaterialModel.AbbeVd,  # noqa: SLF001
        solve_data._S_MaterialModel.dPgF,  # noqa: SLF001
    ), result


def fit_opticstudio_model(
    oss: OpticStudioSystem,
    refractive_indices: dict[float, float],
    *,
    fit_dispersion: bool = False,
) -> tuple[MaterialModel, dict[float, OptimizationResult]]:
    create_model(oss, fit_dispersion=fit_dispersion)

    zp.analyses.systemviewers.CrossSection().run(oss)

    build_merit_function(oss, refractive_indices)

    return run_optimization(oss)


def fit_optiland_model(
    refractive_indices: dict[float, float], *, fit_dispersion: bool = False
) -> tuple[MaterialModel, dict[float, OptimizationResult]]:
    if fit_dispersion:
        raise NotImplementedError("Partial dispersion fitting is not supported in OptiLand.")

    def model(wavelength, n, abbe):
        return AbbeMaterial(n, abbe).n(wavelength)

    popt, _ = curve_fit(
        model,
        list(refractive_indices.keys()),
        list(refractive_indices.values()),
        p0=(1, 0),
    )

    fit_model = AbbeMaterial(popt[0], popt[1])

    return MaterialModel(*popt), {w: OptimizationResult(n, fit_model.n(w)) for w, n in refractive_indices.items()}


def opticstudio_calculate_refractive_indices(
    oss: OpticStudioSystem,
    wavelengths: np.ndarray | list[float],
    material_model: MaterialModel,
    surface: int = 2,
) -> np.ndarray:
    create_model(oss)
    zp.solvers.material_model(
        oss.LDE.GetSurfaceAt(surface).MaterialCell,
        refractive_index=material_model.refractive_index,
        abbe_number=material_model.abbe_number,
        partial_dispersion=material_model.partial_dispersion,
    )

    oss.MFE.DeleteAllRows()

    wavelength_number = oss.SystemData.Wavelengths.NumberOfWavelengths + 1
    oss.SystemData.Wavelengths.AddWavelength(wavelengths[0], 1)

    operand = oss.MFE.GetOperandAt(1)
    operand.ChangeType(zp.constants.Editors.MFE.MeritOperandType.INDX)
    operand.GetCellAt(2).Value = str(surface)  # Surface number
    operand.GetCellAt(3).Value = str(wavelength_number)  # Wavelength number

    refractive_indices = np.zeros(len(wavelengths))

    for i, wavelength in enumerate(wavelengths):
        oss.SystemData.Wavelengths.GetWavelength(wavelength_number).Wavelength = wavelength

        oss.MFE.CalculateMeritFunction()
        refractive_indices[i] = operand.Value

    return refractive_indices


def optiland_calculate_refractive_indices(
    wavelengths: np.ndarray | list[float], material_model: MaterialModel
) -> np.ndarray:
    optiland_model = AbbeMaterial(material_model.refractive_index, material_model.abbe_number)

    return np.array([optiland_model.n(wavelength) for wavelength in wavelengths])


def plot_refractive_indices(
    oss: OpticStudioSystem,
    wavelengths: np.ndarray,
    material_models: dict[str, MaterialModel],
) -> None:
    fig, ax = plt.subplots()

    for (structure, material_model), color in zip(material_models.items(), TABLEAU_COLORS, strict=False):
        refractive_indices_opticstudio = opticstudio_calculate_refractive_indices(oss, wavelengths, material_model)
        refractive_indices_optiland = optiland_calculate_refractive_indices(wavelengths, material_model)

        ax.plot(
            wavelengths,
            refractive_indices_opticstudio,
            color=color,
            label=f"{structure}",
        )
        ax.plot(wavelengths, refractive_indices_optiland, color=color, linestyle="--")

        for wavelength, model in NAVARRO_MATERIAL_MODELS.items():
            refractive_index = getattr(model, structure).refractive_index
            ax.scatter([wavelength], [refractive_index], color=color, marker="x")

    ax.set_xlabel("Wavelength [μm]")
    ax.set_ylabel("Refractive index")
    box = ax.get_position()
    ax.set_position((box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85))

    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([], [], color="black", label="OpticStudio"),
            Line2D([], [], color="black", linestyle="--", label="Optiland"),
        ]
    )
    labels.extend(
        [
            "OpticStudio",
            "Optiland",
        ]
    )
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    # plt.tight_layout()
    plt.show()


def save_fit_results(
    material_models: dict[str, MaterialModel],
    optimization_results: dict[str, dict[float, OptimizationResult]],
    filename: str | Path,
) -> None:
    filename = Path(filename).resolve()
    if not filename.parent.exists():
        print(f"Cannot create output file, {filename.parent} does not exist.")
        sys.exit(1)

    result = {
        "models": {
            structure: {
                "refractive_index": model.refractive_index,
                "abbe_number": model.abbe_number,
                "partial_dispersion": model.partial_dispersion,
            }
            for structure, model in material_models.items()
        },
        "mean_absolute_errors": {
            structure: np.mean(np.abs([result.difference for result in results.values()]))
            for structure, results in optimization_results.items()
        },
    }

    filename.write_text(json.dumps(result, indent=4), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    # Initialize OpticStudio
    zos = zp.ZOS()
    oss = zos.connect("standalone")

    material_models: dict[str, MaterialModel] = {}
    optimization_results: dict[str, dict[float, OptimizationResult]] = {}

    for structure in STRUCTURES:
        refractive_indices = {
            wavelength: getattr(model, structure).refractive_index
            for wavelength, model in NAVARRO_MATERIAL_MODELS.items()
        }

        match args.backend:
            case "optiland":
                material_model, optimization_result = fit_optiland_model(
                    refractive_indices, fit_dispersion=args.fit_dispersion
                )
            case "opticstudio":
                material_model, optimization_result = fit_opticstudio_model(
                    oss, refractive_indices, fit_dispersion=args.fit_dispersion
                )
            case _:
                print(f"Unknown backend: {args.backend}")
                sys.exit(1)

        material_models[structure] = material_model
        optimization_results[structure] = optimization_result

        if args.fit_dispersion:
            print(
                f"{structure}: n = {material_model.refractive_index:.4f}, "
                f"Vd = {material_model.abbe_number:.4f}, "
                f"dPgF = {material_model.partial_dispersion:.4f}"
            )
        else:
            print(f"{structure}: n = {material_model.refractive_index:.4f}, Vd = {material_model.abbe_number:.4f}")

        for wavelength, result in optimization_result.items():
            print(
                f"Wavelength: {wavelength:6f}, Target: {result.target:.4f}, Value: {result.value:.4f}, Difference: {result.difference:+.4g}"
            )

        print("\n")

    if args.output:
        save_fit_results(material_models, optimization_results, args.output)
    else:
        filename = f"material_models_{args.backend}"
        if args.fit_dispersion:
            filename += "_dispersion"

        save_fit_results(
            material_models,
            optimization_results,
            Path(__file__).parent / f"{filename}.json",
        )

    wavelengths = np.linspace(0.4, 0.7, 50)  # μm
    plot_refractive_indices(oss, wavelengths, material_models)


if __name__ == "__main__":
    # Restore default rcparams
    plt.rcdefaults()

    parser = argparse.ArgumentParser(description="Fit material models to refractive indices.")
    parser.add_argument(
        "--backend",
        choices=["optiland", "opticstudio"],
        default="opticstudio",
        help="Backend to use for fitting",
    )
    parser.add_argument(
        "--fit-dispersion",
        action="store_true",
        default=False,
        help="Fit dispersion (only supported in OpticStudio)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output file for fit results")

    args = parser.parse_args()
    main(args)
