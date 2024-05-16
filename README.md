# EyeSimulator

This project aims to provide a standardized method to perform optical (ray tracing) simulations on eye models. `EyeSimulator` is just a working name and should preferrably be replaced with something better before releasing this library.

## Goals

1. Provide a uniform interface to define, build and analyze various types of eye models, using abstractions that make sense in a clinical context;
2. Provide an accessible interface to the most common analyses on these models;
3. Decouple the model definition interface as much as possible from the simulation software, i.e. Zemax OpticStudio;
4. Introduce this as a standardized method for optical simulations of the eye, that can be used by the broader physiological optics community.

## Current implementation

In its current state, `EyeSimulator` provides

- Classes to define models in terms of clinically relevant and measurable parameters (`EyeGeometry`);
- An abstract base class `BaseEye` and a simple implementation `Eye`, which allows to build and analyze these eye models;
- A `Surface` class, which acts as a bridge between the abstractions used in this library and Zemax OpticStudio. This allows for seamless integration of OpticStudio models in our `Eye` class.

## Example

```python
import visisipy
import matplotlib.pyplot as plt

# Initialize the default Navarro model
model = visisipy.EyeModel()

# Build the model in OpticStudio
model.build()

# Visualize the model
fig, ax = plt.subplots()
visisipy.plots.plot_eye(ax, model.geometry, lens_edge_thickness=0.5)
ax.set_xlim((-7, 23))
ax.set_ylim((-15, 15))
ax.set_aspect('equal')
plt.show()
```

```python
import zospy as zp
from visisipy import EyeModel, NavarroGeometry
from visisipy.opticstudio import OpticStudioEye

# Initialize ZOSPy
zos = zp.ZOS()
oss = zos.connect()

# Initialize an eye, using a slightly modified Navarro model
eye_model = EyeModel(NavarroGeometry(iris_radius=0.5))
eye = OpticStudioEye(eye_model)

# Update a parameter of one of the eye's surfaces
eye.lens_front.refractive_index += 1.0

# Build the eye in OpticStudio
eye.build(oss)

# Change the lens's refractive index back
eye.lens_front.refractive_index -= 1.0

# Insert a new (dummy) surface in OpticStudio
oss.LDE.InsertNewSurfaceAt(1).Comment = "input beam"

# Relink the surfaces of the eye, because a new surface was added
eye.relink_surfaces(oss)

# Check if the relinking succeeded
assert eye.lens_back.radius == eye_model.geometry.lens_back_curvature
```

## Future ideas

- Provide (customizable) geometry definitions for various standard eye models, e.g. `NavarroGeometry`, `GullstrandGeometry`.
    - Most likely as child classes of `EyeGeometry`. The default values currently used in `EyeGeometry` will then be removed.
    - The same applies for the `EyeMaterials` class.
- Add more implementations of `BaseEye`, e.g. `ReverseEye`, a reversed version of `Eye`.
- Integrate the eye plot functions defined in [chaasjes/utilities](https://git.lumc.nl/chaasjes/utilities/-/tree/main/utilities/plots/eye) in this library for easy visualization of eye models.