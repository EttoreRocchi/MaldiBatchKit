"""Sphinx configuration for MaldiBatchKit documentation."""

import os
import shutil
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath("../.."))

from maldibatchkit import __version__  # noqa: E402

# Copy notebooks from repo root into the Sphinx source tree so that nbsphinx
# processes real files rather than following a symlink.  Symlinked notebooks
# break image extraction on ReadTheDocs.
_here = Path(__file__).parent
_notebooks_src = _here.parent.parent / "notebooks"
_notebooks_dst = _here / "tutorials" / "notebooks"

if _notebooks_src.exists():
    if _notebooks_dst.is_symlink():
        _notebooks_dst.unlink()
    if _notebooks_dst.exists():
        shutil.rmtree(_notebooks_dst)
    shutil.copytree(
        _notebooks_src,
        _notebooks_dst,
        ignore=shutil.ignore_patterns("__pycache__", "_demo.py", "*.py"),
    )

# Expose a sphinx-click-compatible Click object for the Typer app so that
# the ``.. click::`` directive in cli.rst can introspect the full command
# tree without us having to modify the package source.
try:
    import typer

    from maldibatchkit import cli as _cli_module

    if not hasattr(_cli_module, "typer_click_object"):
        _cli_module.typer_click_object = typer.main.get_command(_cli_module.app)
except Exception:  # pragma: no cover - documentation-only shim
    pass

# Project information
project = "MaldiBatchKit"
copyright = "2026, Ettore Rocchi"
author = "Ettore Rocchi"

# The full version, including alpha/beta/rc tags
release = __version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "sphinx_design",
    "sphinx_click",
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Autosummary settings
autosummary_generate = True

# Type hints settings
typehints_fully_qualified = False
always_document_param_types = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "maldiamrkit": ("https://maldiamrkit.readthedocs.io/en/latest/", None),
    "combatlearn": ("https://combatlearn.readthedocs.io/en/latest/", None),
}

# Suppress warnings for shorthand types in NumPy-style docstrings
# that Sphinx cannot resolve
nitpick_ignore_regex = [
    (r"py:class", r"optional"),
    (r"py:class", r"default=.*"),
    (r"py:class", r"array-like"),
    (r"py:class", r"np\..*"),
    (r"py:class", r"pd\..*"),
    (r"py:class", r"ndarray"),
    (r"py:class", r"Path"),
    (r"py:class", r"arrays"),
    (r"py:class", r"tuples"),
    (r"py:class", r"callable"),
    (r"py:class", r"ignored"),
    (r"py:class", r"self"),
    (r"py:class", r"transformer"),
    (r"py:class", r"typing\..*"),
    (r"py:class", r"ArrayLike"),
    (r"py:class", r"matplotlib\..*"),
    (r"py:class", r"pandas\..*"),
    (r"py:class", r"umap\..*"),
    (r"py:class", r"MaldiSet"),
    (r"py:class", r"maldiamrkit\..*"),
    (r"py:class", r"maldibatchkit\..*"),
    (r"py:class", r"combatlearn\..*"),
    (r"py:class", r"\d+"),
    (r"py:class", r"\{.*"),
    (r"py:class", r"\".*"),
    (r"py:data", r"typing\..*"),
]

# Static files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "_static/maldibatchkit_logo.png"

html_theme_options = {
    # Logo configuration
    "logo": {
        "text": "MaldiBatchKit",
        "image_light": "_static/maldibatchkit_logo.png",
        "image_dark": "_static/maldibatchkit_logo.png",
    },
    # Top navigation bar layout
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "header_links_before_dropdown": 4,
    # Icon links (GitHub, PyPI)
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/EttoreRocchi/MaldiBatchKit",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/MaldiBatchKit/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    # Sidebar behaviour
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 1,
    "collapse_navigation": True,
    # Footer
    "footer_start": ["copyright"],
    "footer_end": ["last-updated"],
    # Syntax highlighting
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}

# Sidebar configuration: no left sidebar on the landing page
html_sidebars = {
    "**": ["sidebar-nav-bs"],
    "index": [],
}

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
