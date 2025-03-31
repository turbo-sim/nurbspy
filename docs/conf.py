import os
import sys

# Add package root dir to path
sys.path.insert(0, os.path.abspath(".."))

# Define project metadata
project = "nurbspy"
copyright = ""
author = "Roberto Agromayor"
release = "v1.1.4"

# Define extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "numpydoc",
    "sphinx.ext.todo",
    # 'sphinx_tabs.tabs',
    "sphinx_togglebutton",
    "sphinx_design",
]

# Avoid warnings when generating the summary table of class methods
autosummary_generate = True
numpydoc_class_members_toctree = False

todo_include_todos = True

# Add bibliography file
bibtex_bibfiles = ["source/bibliography.bib"]
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"

# Exclude unnecessary files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# exclude_patterns.extend(["source/api/barotropy.isentropic_nozzle.rst", "source/api/barotropy.sCO2_utilities.rst"])
exclude_patterns.extend(["source/api/modules.rst"])
# exclude_patterns.extend(["source/api/barotropy.fluid_properties.fluid_properties.rst", "source/api/barotropy.fluid_properties.core_calculations.rst"])


# html_theme_options = {
#     "show_toc_level": 2,  # Level of indentation on side panel TOC
# }

# Define theme
html_theme = "sphinx_book_theme"
# html_theme = 'pydata_sphinx_theme'
# html_theme = 'sphinx_rtd_theme'


