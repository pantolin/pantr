"""Sphinx configuration for PaNTr documentation.
Initializes metadata, extensions, and build parameters."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Final

import sys
from sphinx.util import logging

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
SRC_PATH: Final[Path] = PROJECT_ROOT / "src"
DOXYGEN_XML_DIR: Final[Path] = PROJECT_ROOT / "build" / "doxygen" / "xml"
LOGGER = logging.getLogger(__name__)

sys.path.insert(0, str(SRC_PATH))

import pantr
CURRENT_YEAR: Final[int] = date.today().year

project = "PaNTr"
author = "Pablo Antolin"
copyright = (
    f"{CURRENT_YEAR}, Pablo Antolin"
)  # pylint: disable=redefined-builtin

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "breathe",
    "jupytext.sphinx",
    "sphinx_rtd_dark_mode",
]

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

breathe_projects: dict[str, str] = {}
if DOXYGEN_XML_DIR.exists():
    breathe_projects["pantr"] = str(DOXYGEN_XML_DIR)
else:
    LOGGER.warning("Doxygen XML directory %s not found; skipping C++ API import.", DOXYGEN_XML_DIR)
breathe_default_project = "pantr"

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

