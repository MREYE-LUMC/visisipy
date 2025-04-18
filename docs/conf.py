import os
from datetime import datetime

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Visisipy"
year = datetime.now().year  # noqa: DTZ005
copyright_year = str(year) if year == 2025 else f"2025 - {year}"
copyright = f"{copyright_year}, Corné Haasjes, Luc van Vught, Jan-Willem Beenakker"  # noqa: A001
author = "Corné Haasjes, Luc van Vught, Jan-Willem Beenakker"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["autoapi.extension", "myst_nb", "sphinx_design"]
myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MREYE-LUMC/visisipy",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/visisipy",
            "icon": "fa-brands fa-python",
        },
    ],
    "logo": {
        "text": "Visisipy",
    },
    "show_toc_level": 1,
}

html_context = {
    "github_user": "MREYE-LUMC",
    "github_repo": "visisipy",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Options for myst-nb ----------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html

nb_execution_mode = "off" if os.getenv("READTHEDOCS") or os.getenv("BUILD_NOTEBOOKS") == "false" else "cache"

# -- Options for autoapi ----------------------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/

autoapi_dirs = ["../visisipy"]
autoapi_root = "api"
autoapi_options = [
    "members",
    # "undoc-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
    # "special-members",
    # "imported-members",
]
