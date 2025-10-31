# Installation

::::{tab-set}
:sync-group: installation

:::{tab-item} uv
:sync: uv

We recommend installing Visisipy in a separate environment with a project manager like [`uv`](https://docs.astral.sh/uv/):

```bash
uv add visisipy
```
:::

:::{tab-item} pip
:sync: pip

Visisipy is available on [PyPI](https://pypi.org/project/visisipy/) and can be installed through `pip`:

```bash
pip install visisipy
```
:::

:::{tab-item} conda
:sync: conda

Visisipy is available on [conda-forge](https://anaconda.org/conda-forge/visisipy) and can be installed through `conda`:

```bash
conda install -c conda-forge visisipy
```
:::

::::

## Enabling GPU Acceleration

To use Torch for GPU-accelerated ray tracing, you need to install the `torch` package:

::::{tab-set}
:sync-group: installation

:::{tab-item} uv
:sync: uv

```bash
uv add visisipy[torch]
```
:::

:::{tab-item} pip
:sync: pip

```bash
pip install visisipy[torch]
```
:::

:::{tab-item} conda
:sync: conda

```bash
conda install pytorch
```
:::

::::

GPU acceleration is only available for the [Optiland backend](#optiland-gpu).
