# Simple example

The example below shows how to create a custom eye model from clinical parameters, build it in OpticStudio, and interact with the model.

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