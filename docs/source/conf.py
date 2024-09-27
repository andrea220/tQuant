import sphinx_rtd_theme
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Modifica il percorso secondo la struttura del tuo progetto
sys.path.insert(0, os.path.abspath('../'))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TensorQuant'
copyright = '2024, Andrea Carapelli'
author = 'Andrea Carapelli'
release = '0.0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Per documentazione automatica
    'sphinx.ext.napoleon',  # Per supportare le docstring in stile Google/Numpy
    'sphinx.ext.viewcode',
]
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

autoapi_type = 'python'
autoapi_options = [
    "members",
    "undoc-members",
    "inherited-members",
    "special-members",
    "show-inheritance",
    "show-module-summary",
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 4,  # Profondità di navigazione nel menu laterale
    'collapse_navigation': False,  # Non comprimere la navigazione nel menu
    'style_external_links': True,  # Stile per i link esterni
}
