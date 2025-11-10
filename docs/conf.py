"""Sphinx configuration for PaNTr documentation.
Initializes metadata, extensions, and build parameters."""

from __future__ import annotations

import importlib.util
import warnings
import sys
from datetime import date
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
SRC_PATH: Final[Path] = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

pantr_spec = importlib.util.spec_from_file_location(
    "pantr", SRC_PATH / "pantr" / "__init__.py"
)
if pantr_spec is None or pantr_spec.loader is None:
    msg = f"Unable to locate pantr package at {SRC_PATH / 'pantr' / '__init__.py'}"
    raise ImportError(msg)
pantr = importlib.util.module_from_spec(pantr_spec)
pantr_spec.loader.exec_module(pantr)
CURRENT_YEAR: Final[int] = date.today().year

project = "PaNTr"
author = "Pablo Antolin"
copyright = f"{CURRENT_YEAR}, Pablo Antolin"  # pylint: disable=redefined-builtin

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "jupytext.sphinx",
    "sphinx_rtd_dark_mode",
]

JUPYTEXT_EXTENSION: Final[str] = "jupytext.sphinx"
if importlib.util.find_spec(JUPYTEXT_EXTENSION) is None:
    warnings.warn(
        f"Skipping optional Sphinx extension {JUPYTEXT_EXTENSION!r}: module not found.",
        stacklevel=1,
    )
    extensions = [ext for ext in extensions if ext != JUPYTEXT_EXTENSION]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable", {}),
    "scipy": ("https://docs.scipy.org/doc/scipy/", {}),
    "matplotlib": ("https://matplotlib.org/stable", {}),
}

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "linkify",
    "smartquotes",
]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_show_sourcelink = True

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
}

html_context = {
    "display_github": True,
    "github_user": "pablodroca",
    "github_repo": "pantr",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_logo = None
html_favicon = None

pygments_style = "default"
pygments_dark_style = "native"

version = pantr.__version__
release = pantr.__version__

nitpicky = True
