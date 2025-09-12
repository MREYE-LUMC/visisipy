from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import CenteredNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import visisipy
from visisipy.backend import BackendSettings
from visisipy.opticstudio.backend import OpticStudioBackend
from visisipy.optiland.backend import OptilandBackend


def plot_dataframe(ax: plt.Axes, df: pd.DataFrame, title: str, cbar_label: str = "Relative intensity", **kwargs):
    im = ax.imshow(
        df.values,
        extent=(df.columns[0], df.columns[-1], df.index[0], df.index[-1]),
        origin="lower",
        **kwargs,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    ax.set_title(title)
    ax.set_xlabel("X [μm]")
    ax.set_ylabel("Y [μm]")

    plt.colorbar(im, label=cbar_label, cax=cax)


def compare_opd_maps(model: visisipy.EyeModel, field: tuple[float, float], sampling: int = 128):
    """
    Compare OPD maps for OpticStudio and Optiland backends.
    """
    opd_map_opticstudio = visisipy.analysis.opd_map(
        model=model,
        field_coordinate=field,
        sampling=sampling,
        remove_tilt=False,
        backend=OpticStudioBackend,
    )

    opd_map_optiland = visisipy.analysis.opd_map(
        model=model,
        field_coordinate=field,
        sampling=sampling - 1,
        remove_tilt=False,
        backend=OptilandBackend,
    )

    difference = pd.DataFrame(
        opd_map_opticstudio.values[1:, 1:] - opd_map_optiland.values,
        index=opd_map_optiland.index,
        columns=opd_map_optiland.columns,
    )

    fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    plot_dataframe(ax[0], opd_map_opticstudio, "OpticStudio OPD map", cbar_label="Difference [Waves]")
    plot_dataframe(ax[1], opd_map_optiland, "Optiland OPD map", cbar_label="Difference [Waves]")
    plot_dataframe(ax[2], difference, "Difference", "OpticStudio - Optiland", cmap="coolwarm", norm=CenteredNorm())

    fig.suptitle(f"OPD map Comparison for x={field[0]}°, y={field[1]}°")


BACKEND_SETTINGS = BackendSettings(
    field_type="angle",
    fields=[(0, 0)],
    wavelengths=[0.543],
    aperture_type="float_by_stop_size",
    aperture_value=3.0,
)

OpticStudioBackend.initialize(**BACKEND_SETTINGS, ray_aiming="off")
OptilandBackend.initialize(**BACKEND_SETTINGS)

# Create an eye model in OpticStudio
model = visisipy.EyeModel(geometry=visisipy.models.NavarroGeometry(), materials=visisipy.models.NavarroMaterials543())

OpticStudioBackend.build_model(model)
OptilandBackend.build_model(model)

compare_opd_maps(model, field=(0, 30))
plt.show()
