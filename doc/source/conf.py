# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SOFC multiscale'
# copyright = '2024, Graziella'
author = 'Hamid Abbasi'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "jupyter_sphinx",
    "nbsphinx",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
# html_theme = "classic"
# html_theme_options = {
#     "rightsidebar": "true",
#     "relbarbgcolor": "black"
# }
html_static_path = ['_static']
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        # 'searchbox.html',
    ]
}
html_theme_options = {
    'sidebar_width': '300px',
    'fixed_sidebar': True,
}