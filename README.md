# Visisipy: accessible vision simulations in Python

Visisipy (pronounced `/ˌvɪsəˈsɪpi/`, like Mississippi but with a V) is a Python library for optical simulations of the eye.
It provides an easy-to-use interface to define and build eye models, and to perform common ophthalmic analyses on these models.

## Goals

1. Provide a uniform interface to define, build and analyze various types of eye models, using abstractions that make sense in a clinical context;
2. Provide a collection of ready-to-use eye models, such as the Navarro model, that can be customized at need;
3. Provide an accessible interface to clinically relevant analyses on these models.

All calculations are currently performed in OpticStudio through the [ZOSPy][zospy] library, but visisipy is designed in a
modular fashion to allow for other backends in the future.

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
raytrace = visisipy.analysis.raytrace(coordinates=zip([0] * 5, range(0, 60, 10)))

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

## Planned functions

- Generation of realistic randomized eye models using the method proposed by Rozema et al.;

## Future ideas

- Provide (customizable) geometry definitions for other standard eye models, e.g. `GullstrandGeometry`.
- Add support for reversed eyes.

[zospy]: https://zospy.readthedocs.io/