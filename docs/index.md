# Welcome to Visisipy's documentation!

[![PyPI - Version](https://img.shields.io/pypi/v/visisipy)](https://pypi.org/project/visisipy)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FMREYE-LUMC%2Fvisisipy%2Fmain%2Fpyproject.toml)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/MREYE-LUMC/visisipy/ci.yml)

Visisipy (pronounced `/ˌvɪsəˈsɪpi/`, like Mississippi but with a V) is a Python library for optical simulations of the eye.
It provides an easy-to-use interface to define and build eye models, and to perform common ophthalmic analyses on these models.

## Project goals

::::{grid} 1 1 2 2
:gutter: 4

:::{grid-item-card}
{fa}`eye;pst-color-primary` **Uniform model definitions**
^^^
Provide a uniform interface to define, build and analyze various types of eye models, using abstractions that are relevant in a clinical context.
:::

:::{grid-item-card} 
{fa}`box-open;pst-color-primary` **Ready-to-use models**
^^^
Provide a collection of ready-to-use eye models, such as the Navarro model[^navarro], that can be customized at need.
:::

:::{grid-item-card}
{fa}`chart-line;pst-color-primary` **Accessible analyses**
^^^
Provide an accessible interface to clinically relevant analyses on these models, such as [off-axis refraction calculations][refraction].
:::

:::{grid-item-card}
{fa}`puzzle-piece;pst-color-primary` **Multiple backends**
^^^
Modular design with support for [multiple backends][backends], both open-source and commercial.
:::
::::

## User guide

```{toctree}
:maxdepth: 2

user_guide/index
```

## Examples

```{toctree}
:maxdepth: 2

examples/index
```

## Contributing

```{toctree}
:maxdepth: 2

Contributing <contributing>
```

## API

```{toctree}
:maxdepth: 2

API <api/index>
```

[zospy]: https://zospy.readthedocs.io/
[opticstudio]: https://www.ansys.com/products/optics/ansys-zemax-opticstudio
[optiland]: https://optiland.readthedocs.io/
[backends]: user_guide/backend.ipynb
[refraction]: user_guide/analyses.ipynb#refraction

[^navarro]: Escudero-Sanz, I., & Navarro, R. (1999). Off-axis aberrations of a wide-angle schematic eye model. JOSA A, 16(8), 1881–1891. https://doi.org/10.1364/JOSAA.16.001881
<!-- [^rozema]: Rozema, J. J., Rodriguez, P., Navarro, R., & Tassignon, M.-J. (2016). SyntEyes: A Higher-Order Statistical Eye Model for Healthy Eyes. Investigative Ophthalmology & Visual Science, 57(2), 683–691. https://doi.org/10.1167/iovs.15-18067 -->
[^zospy]: Vught, L. van, Haasjes, C., & Beenakker, J.-W. M. (2024). ZOSPy: Optical ray tracing in Python through OpticStudio. Journal of Open Source Software, 9(96), 5756. https://doi.org/10.21105/joss.05756
