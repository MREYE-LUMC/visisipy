# Contributing to Visisipy

Thank you for your interest in contributing to Visisipy!
Visisipy aims to be a community-driven project and warmly accepts contributions.
Please follow the guidelines below to ensure a smooth collaboration.

## Contribution ideas

- [Add an example](#adding-examples) demonstrating a specific use case of Visisipy.
- [Add a new analysis](#writing-analyses).
- [Extend the documentation](#writing-documentation) with additional information or examples.

## How to contribute

1. **Discuss your idea**  
   Before starting work on a new feature or significant change, please [open a discussion] to share your idea. 
   This helps us ensure alignment with the project's goals and avoids duplicate efforts.

2. **Fork and clone the repository**  
   Fork the repository to your GitHub account and clone it locally:
   ```bash
   git clone https://github.com/<your-username>/visisipy.git
   cd visisipy
   ```
   
3. **Set up the development environment**
   Visisipy uses [`hatch`][hatch] to manage the development environment.
   To create a virtual environment for development, run:
   ```bash
    hatch env create
    ```
   
4. **Make your changes**  
   Make your changes in the codebase. Please ensure that your code follows the project's [style guidelines](#code-style).
   
5. **Test your changes**
    Run the tests to ensure that everything works as expected:
    ```bash
    hatch test
    ```
    If you are adding new features or fixing bugs, please add corresponding tests.
    Before you open a Pull Request, run the tests for all supported Python versions:
    ```bash
    hatch test -a
    ```
   
6. **Contribute your changes**  
   Once you are satisfied with your changes, push them to your forked repository and open a Pull Request against the main repository.
   Please provide a clear description of your changes and reference any related issues or discussions.

## Code style

Visisipy uses `ruff` to format code and check for common issues.
To format your code, run:

```bash
hatch fmt
```

Before opening a Pull Request, please ensure that you have fixed all issues reported by `ruff`.
Docstrings should be formatted according to [numpydoc].
Compliance with numpydoc can be checked using `pydocstringformatter`:

```bash
hatch run format-docstrings
```

Contributions will only be accepted if they follow these guidelines.

## Writing documentation

The documentation is built using [Sphinx], the [PyData Sphinx Theme] and [MyST-NB].
and is located in the `docs` directory.
To build the documentation, run:

```bash
hatch run docs:build
```

To get a live preview of the documentation that is updated automatically, run:

```bash
hatch run docs:preview
```

## Adding examples

Please contribute examples as Jupyter notebooks and place them in a subdirectory of the `docs/examples` directory.
By default, notebooks are executed when building the documentation.
Because it is not possible to run examples that use the OpticStudio backend on ReadTheDocs, please use the Optiland
backend wherever possible.
If your example requires the OpticStudio backend, please add it to the `execution_excludepatterns` variable in `conf.py`,
and mention the reason why the example requires OpticStudio in your Pull Request.

## Writing analyses

When writing a new analysis, make sure it follows the structure documented {py:mod}`here <visisipy.analysis>`.
Analysis functions are decorated with the {py:func}`@analysis <visisipy.analysis.base.analysis>` decorator, which
ensures the required analysis structure is followed.
In principle, the analysis should be implemented for all backends.
If the analysis is only implemented for a specific backend, please mention this in your Pull Request.

[open a discussion]: https://github.com/MREYE-LUMC/visisipy/discussions
[hatch]: https://hatch.pypa.io/latest/
[numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
[Sphinx]: https://www.sphinx-doc.org/en/master/
[PyData Sphinx Theme]: https://pydata-sphinx-theme.readthedocs.io/en/latest/
[MyST-NB]: https://myst-nb.readthedocs.io/en/latest/
