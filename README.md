# Visisipy: accessible vision simulations in Python

Visisipy (pronounced `/ˌvɪsəˈsɪpi/`, like Mississippi but with a V) is a Python library for optical simulations of the eye.
It provides an easy-to-use interface to define and build eye models, and to perform common ophthalmic analyses on these models.

## Goals

1. Provide a uniform interface to define, build and analyze various types of eye models, using abstractions that are relevant in a clinical context.
2. Provide a collection of ready-to-use eye models, such as the Navarro model[^navarro], that can be customized at need.
3. Provide an accessible interface to clinically relevant analyses on these models, such as off-axis refraction calculations.
4. Modular design with support for multiple backends, both open-source and commercial.

## Contributing

Visisipy aims to be a community-driven project and warmly accepts contributions.
If you want to contribute, please email us (visisipy@mreye.nl) or [open a new discussion](https://github.com/MREYE-LUMC/visisipy/discussions).

## Installation

Visisipy can be installed through `pip`:

```bash
pip install visisipy
```

Visisipy will be made available through Conda as soon as possible.

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

## Documentation

Read the full documentation at [visisipy.readthedocs.io](https://visisipy.readthedocs.io).

[//]: # (References)
[^navarro]: Escudero-Sanz, I., & Navarro, R. (1999). Off-axis aberrations of a wide-angle schematic eye model. JOSA A, 16(8), 1881–1891. https://doi.org/10.1364/JOSAA.16.001881