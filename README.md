# Visisipy: accessible vision simulations in Python

Visisipy (pronounced `/ˌvɪsəˈsɪpi/`, like Mississippi but with a V) is a Python library for optical simulations of the eye.
It provides an easy-to-use interface to define and build eye models, and to perform common ophthalmic analyses on these models.

## Goals

1. Provide a uniform interface to define, build and analyze various types of eye models, using abstractions that make sense in a clinical context;
2. Provide a collection of ready-to-use eye models, such as the Navarro model[^navarro], that can be customized at need;
3. Provide an accessible interface to clinically relevant analyses with these models.

All calculations are currently performed in OpticStudio through the [ZOSPy][zospy] library[^zospy], but visisipy is designed in a modular fashion to allow for other backends in the future.

## Contributing

Visisipy aims to be a community-driven project and warmly accepts contributions.
If you want to contribute, please email us (visisipy@mreye.nl) or [open a new discussion](https://github.com/MREYE-LUMC/visisipy/discussions).

## Installation

Visisipy can be installed through `pip`:

```bash
pip install git+https://github.com/MREYE-LUMC/visisipy.git
```

Visisipy will be made available through PyPI and Conda as soon as possible.

## Example

```python
import visisipy
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the default Navarro model
model = visisipy.EyeModel()

# Build the model in OpticStudio
model.build()

# Perform a raytrace analysis
coordinates = [(0, 0), (0, 10), (0, 20), (0, 30), (0, 40)]
raytrace = visisipy.analysis.raytrace(coordinates=coordinates)

# Alternatively, the model can be built and analyzed in one go:
# raytrace = visisipy.analysis.raytrace(model, coordinates=zip([0] * 5, range(0, 60, 10)))

# Visualize the model
fig, ax = plt.subplots()
visisipy.plots.plot_eye(ax, model.geometry, lens_edge_thickness=0.5)
ax.set_xlim((-7, 23))
ax.set_ylim((-15, 15))
ax.set_aspect('equal')

sns.lineplot(raytrace, x="z", y="y", hue="field", ax=ax)

plt.show()
```

### Configure the backend

Visisipy uses OpticStudio as a backend for calculations; this is currently the only supported backend.
This backend is automatically started and managed in the background, but can also be configured manually.

```python
import visisipy

# Use OpticStudio in standalone mode (default)
visisipy.set_backend("opticstudio")

# Use OpticStudio in extension mode
visisipy.set_backend("opticstudio", mode="extension")

# Use OpticStudio in extension mode with ray aiming enabled
visisipy.set_backend("opticstudio", mode="extension", ray_aiming="real")

# Get the OpticStudioSystem from visisipy to interact with it manually
# This only works when the backend is set to "opticstudio"
# See https://zospy.readthedocs.io/en/latest/api/zospy.zpcore.OpticStudioSystem.html for documentation of this object
oss = visisipy.backend.get_oss()
```

### Create a custom eye model from clinical parameters

An eye model in visispy consists of two parts: the geometry and the material properties.
The geometry is defined by `visisipy.models.EyeGeometry`, and the material properties are defined by `visisipy.models.Materials`.
They are combined in `visisipy.EyeModel` to constitute a complete eye model.

```python
import visisipy

geometry = visisipy.models.create_geometry(
    axial_length=20,
    cornea_thickness=0.5,
    anterior_chamber_depth=3,
    lens_thickness=4,
    cornea_front_radius=7,
    cornea_front_asphericity=0,
    cornea_back_radius=6,
    cornea_back_asphericity=0,
    lens_front_radius=10,
    lens_front_asphericity=0,
    lens_back_radius=-6,
    lens_back_asphericity=0,
    retina_radius=-12,
    retina_asphericity=0,
    pupil_radius=1.0,
)

# Use this geometry together with the refractive indices of the Navarro model
model = visisipy.EyeModel(geometry=geometry, materials=visisipy.models.materials.NavarroMaterials())

# NavarroMaterials is the default, so this is equivalent:
model = visisipy.EyeModel(geometry=geometry)
```

### Interact with the eye model in OpticStudio

```python
import visisipy

# Just use the default Navarro model
model = visisipy.EyeModel()

# Build the model in OpticStudio
built_model: visisipy.opticstudio.OpticStudioEye = model.build()

# Update the lens front radius
built_model.lens_front.radius = 10.5
```

## Planned functions

- Generation of realistic randomized eye models using the method proposed by Rozema et al.[^rozema]

## Future ideas

- Provide (customizable) geometry definitions for other standard eye models, e.g. `GullstrandGeometry`.
- Add support for reversed eyes.
- Add support for other (open source) ray tracing backends.

[zospy]: https://zospy.readthedocs.io/

[//]: # (References)
[^navarro]: Escudero-Sanz, I., & Navarro, R. (1999). Off-axis aberrations of a wide-angle schematic eye model. JOSA A, 16(8), 1881–1891. https://doi.org/10.1364/JOSAA.16.001881
[^rozema]: Rozema, J. J., Rodriguez, P., Navarro, R., & Tassignon, M.-J. (2016). SyntEyes: A Higher-Order Statistical Eye Model for Healthy Eyes. Investigative Ophthalmology & Visual Science, 57(2), 683–691. https://doi.org/10.1167/iovs.15-18067
[^zospy]: Vught, L. van, Haasjes, C., & Beenakker, J.-W. M. (2024). ZOSPy: Optical ray tracing in Python through OpticStudio. Journal of Open Source Software, 9(96), 5756. https://doi.org/10.21105/joss.05756
