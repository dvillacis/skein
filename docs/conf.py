"""Sphinx config for the skein-glm docs site.

Stack: Sphinx + furo + MyST (markdown via myst-parser) +
sphinx-copybutton + sphinx-design + autodoc + napoleon for
numpy-style docstring parsing + intersphinx for cross-linking
to numpy / scipy / sklearn docs.

Build locally:

    pip install -e ".[docs]"
    sphinx-build -W -b html docs docs/_build/html
    python -m http.server -d docs/_build/html

Read the Docs builds via .readthedocs.yaml (sphinx:configuration =
docs/conf.py).
"""
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

# Make `import skein_glm` importable for autodoc; the package is
# pip-installed via the build-system, but if someone runs sphinx
# without installing first, fall back to the source tree.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

# ---- Project ----------------------------------------------------------------

project = "skein"
author = "David Villacis"
copyright = "2026, David Villacis"

try:
    release = importlib.metadata.version("skein-glm")
except importlib.metadata.PackageNotFoundError:
    # Source-only checkout, no install. Read from pyproject.toml.
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

# ---- General ----------------------------------------------------------------

extensions = [
    # Markdown via MyST. We're markdown-native; no RST except inside
    # `{eval-rst}` blocks for autodoc.
    "myst_parser",
    # Auto API reference from Python docstrings.
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    # Click-to-copy buttons on every code block.
    "sphinx_copybutton",
    # Tab containers, grid cards, dropdowns.
    "sphinx_design",
]

source_suffix = {
    ".md": "myst",
    ".rst": "restructuredtext",
}

# Master document — the root of the toctree.
master_doc = "index"

# Files to exclude from the build.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # The auto-generated build output should never be sphinx'd.
    "**/_build/**",
]

# ---- MyST -------------------------------------------------------------------

myst_enable_extensions = [
    # `$$ ... $$` math blocks + `$x$` inline math.
    "dollarmath",
    "amsmath",
    # `:::{note}` admonitions.
    "colon_fence",
    # GitHub-style tables.
    "deflist",
    "tasklist",
    # Smart quotes / dashes.
    "smartquotes",
    "replacements",
    # Inline emphasis variations.
    "strikethrough",
    # Substitutions like `{{ project }}`.
    "substitution",
]

myst_heading_anchors = 3

# ---- autodoc ----------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    # Don't include private members or sklearn-inherited boilerplate
    # (set_params, get_params, etc.) in the auto-rendered API ref.
    "exclude-members": "__init__, __weakref__",
}

# Numpy-style docstrings — let napoleon parse Parameters/Returns/etc.
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Type hints in signatures.
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"

# ---- intersphinx ------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# ---- Theme: furo ------------------------------------------------------------

html_theme = "furo"
html_title = f"skein {version}"

html_theme_options = {
    "source_repository": "https://github.com/dvillacis/skein/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/dvillacis/skein",
            "html": "",
            "class": "fa-brands fa-solid fa-github fa-2x",
        },
    ],
    # Furo's signature light/dark scheme. Subtle indigo accent.
    "light_css_variables": {
        "color-brand-primary": "#5b21b6",   # deep purple-ish, our brand
        "color-brand-content": "#5b21b6",
    },
    "dark_css_variables": {
        "color-brand-primary": "#a78bfa",
        "color-brand-content": "#a78bfa",
    },
}

# Static assets (CSS overrides, etc.). Empty for now; extend if we
# want a custom logo or tweak.
html_static_path = ["_static"]
templates_path = ["_templates"]

# ---- Misc -------------------------------------------------------------------

# Don't fail the build on missing intersphinx targets — those are
# online lookups and may flake in CI.
nitpicky = False

# Link to the source for each docstring class/function.
viewcode_enable_epub = False
