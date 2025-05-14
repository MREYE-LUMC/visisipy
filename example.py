# %%
from __future__ import annotations

from operator import itemgetter
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import visisipy

sns.set_style(
    "whitegrid",
    rc={
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.color": "gray",
        "grid.alpha": 0.5,
        "patch.edgecolor": "black",
    },
)
# %%
# Initialize the default Navarro model
model = visisipy.EyeModel()

# Build the model in OpticStudio
model.build()

# Add dummy surface
dummy_surface = visisipy.get_backend().oss.LDE.InsertNewSurfaceAt(1)
dummy_surface.Comment = "dummy"
dummy_surface.Thickness = 1

# %% Perform a raytrace analysis
y_angles = range(0, 90, 5)
raytrace = visisipy.analysis.raytrace(coordinates=zip([0] * len(y_angles), y_angles))

raytrace["input_angle"] = raytrace.field.apply(itemgetter(1))


# %% Calculate input and output angles
class InputOutputAngles(NamedTuple):
    input_angle: float
    output_angle_np2: float
    output_angle_pupil: float
    output_angle_retina_center: float

    @staticmethod
    def output_angle(coordinate: tuple[float, float], reference_coordinate: tuple[float, float]) -> float:
        return np.rad2deg(
            np.arctan2(
                coordinate[1] - reference_coordinate[1],
                coordinate[0] - reference_coordinate[0],
            )
        )

    @classmethod
    def from_raytrace(
        cls,
        raytrace: pd.DataFrame,
        location_np2: float,
        location_pupil: float,
        location_retina_center: float,
        coordinate: str = "y",
    ):
        image_coordinate: tuple[float, float] = tuple(raytrace.query("comment == 'retina'").iloc[0][["z", coordinate]])

        return cls(
            input_angle=raytrace.field.iloc[0]["xy".index(coordinate)],
            output_angle_np2=cls.output_angle(image_coordinate, (location_np2, 0)),
            output_angle_pupil=cls.output_angle(image_coordinate, (location_pupil, 0)),
            output_angle_retina_center=cls.output_angle(image_coordinate, (location_retina_center, 0)),
        )


location_np2 = 7.45 - (model.geometry.cornea_thickness + model.geometry.anterior_chamber_depth)
location_retina_center = (
    model.geometry.lens_thickness + model.geometry.vitreous_thickness - abs(model.geometry.retina.half_axes.axial)
)

angles = pd.DataFrame(
    InputOutputAngles.from_raytrace(rt, location_np2, 0, location_retina_center)
    for _, rt in raytrace.groupby("input_angle")
)

# %% Visualize the model and angle relations
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), gridspec_kw={"width_ratios": [1, 0.8]})

sns.lineplot(
    raytrace,
    x="z",
    y="y",
    hue="input_angle",
    legend="brief",
    palette="plasma",
    ax=ax[0],
)
sns.move_legend(ax[0], "lower right", title="input angle [°]")

visisipy.plots.plot_eye(ax[0], model.geometry, lens_edge_thickness=0.5, zorder=5)
ax[0].set_xlim((-7, 23))
ax[0].set_ylim((-15, 15))
ax[0].set_aspect("equal")
ax[0].set_xlabel("z [mm]")
ax[0].set_ylabel("y [mm]")

sns.lineplot(
    data=angles,
    x="input_angle",
    y="output_angle_np2",
    label="second nodal point",
    ax=ax[1],
)
sns.lineplot(
    data=angles,
    x="input_angle",
    y="output_angle_pupil",
    label="pupil",
    ax=ax[1],
)
sns.lineplot(
    data=angles,
    x="input_angle",
    y="output_angle_retina_center",
    label="retina center",
    ax=ax[1],
)
ax[1].set_aspect("equal")
ax[1].set_xlabel("input angle [°]")
ax[1].set_ylabel("output angle [°]")

fig.savefig("visisipy_example.png", dpi=600, bbox_inches="tight")

plt.show()

# %%
