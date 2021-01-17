# pip documentation build configuration file, created by
# sphinx-quickstart on Tue Apr 22 22:08:49 2008
#
# This file is execfile()d with the current directory set to its containing dir
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import glob
import os
import pathlib
import re
import sys

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

docs_dir = os.path.dirname(os.path.dirname(__file__))
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, docs_dir)
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# -- General configuration ----------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
# extensions = ['sphinx.ext.autodoc']
extensions = [
    # native:
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    # third-party:
    'myst_parser',
    'sphinx_inline_tabs',
    'sphinxcontrib.towncrier',
    # in-tree:
    'pip_sphinxext',
]

# intersphinx
intersphinx_cache_limit = 0
intersphinx_mapping = {
    'pypug': ('https://packaging.python.org/', None),
    'pypa': ('https://www.pypa.io/en/latest/', None),
}


# Add any paths that contain templates here, relative to this directory.
templates_path = []

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
# source_encoding = 'utf-8'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'pip'
copyright = '2008-2020, PyPA'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.

version = release = 'dev'

# Readthedocs seems to install pip as an egg (via setup.py install) which
# is somehow resulting in "import pip" picking up an older copy of pip.
# Rather than trying to force RTD to install pip properly, we'll simply
# read the version direct from the __init__.py file. (Yes, this is
# fragile, but it works...)

pip_init = os.path.join(docs_dir, '..', 'src', 'pip', '__init__.py')
with open(pip_init) as f:
    for line in f:
        m = re.match(r'__version__ = "(.*)"', line)
        if m:
            __version__ = m.group(1)
            # The short X.Y version.
            version = '.'.join(__version__.split('.')[:2])
            # The full version, including alpha/beta/rc tags.
            release = __version__
            break

# We have this here because readthedocs plays tricks sometimes and there seems
# to be a heisenbug, related to the version of pip discovered. This is here to
# help debug that if someone decides to do that in the future.
print("pip version:", version)
print("pip release:", release)

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
# unused_docs = []

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_patterns = ['build/']

# The reST default role (used for this markup: `text`) to use for all documents
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

extlinks = {
    'issue': ('https://github.com/pypa/pip/issues/%s', '#'),
    'pull': ('https://github.com/pypa/pip/pull/%s', 'PR #'),
    'pypi': ('https://pypi.org/project/%s/', ''),
}

# Turn off sphinx build warnings because of sphinx tabs during man pages build
sphinx_tabs_nowarn = True

# -- Options for HTML output --------------------------------------------------

# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = "furo"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = f"{project} documentation v{release}"

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = '_static/piplogo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = 'favicon.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, the Docutils Smart Quotes transform (originally based on
# SmartyPants) will be used to convert characters like quotes and dashes
# to typographically correct entities.  The default is True.
smartquotes = True

# This string, for use with Docutils 0.14 or later, customizes the
# SmartQuotes transform. The default of "qDe" converts normal quote
# characters ('"' and "'"), en and em dashes ("--" and "---"), and
# ellipses "...".
#    For now, we disable the conversion of dashes so that long options
# like "--find-links" won't render as "-find-links" if included in the
# text in places where monospaced type can't be used. For example, backticks
# can't be used inside roles like :ref:`--no-index <--no-index>` because
# of nesting.
smartquotes_action = "qe"

# Custom sidebar templates, maps document names to template names.
html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
html_use_modindex = False

# If false, no index is generated.
html_use_index = False

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'pipdocs'


# -- Options for LaTeX output -------------------------------------------------

# The paper size ('letter' or 'a4').
# latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
# latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual])
latex_documents = [
    (
        'index',
        'pip.tex',
        'pip Documentation',
        'pip developers',
        'manual',
    ),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# Additional stuff for the LaTeX preamble.
# latex_preamble = ''

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_use_modindex = True

# -- Options for Manual Pages -------------------------------------------------

# List of manual pages generated
man_pages = [
    (
        'index',
        'pip',
        'package manager for Python packages',
        'pip developers',
        1
    )
]


def to_document_name(path, base_dir):
    """Convert a provided path to a Sphinx "document name".
    """
    relative_path = os.path.relpath(path, base_dir)
    root, _ = os.path.splitext(relative_path)
    return root.replace(os.sep, '/')


# Here, we crawl the entire man/commands/ directory and list every file with
# appropriate name and details
man_dir = os.path.join(docs_dir, 'man')
raw_subcommands = glob.glob(os.path.join(man_dir, 'commands/*.rst'))
if not raw_subcommands:
    raise FileNotFoundError(
        'The individual subcommand manpages could not be found!'
    )
for fname in raw_subcommands:
    fname_base = to_document_name(fname, man_dir)
    outname = 'pip-' + fname_base.split('/')[1]
    description = 'description of {} command'.format(
        outname.replace('-', ' ')
    )

    man_pages.append((fname_base, outname, description, 'pip developers', 1))

# -- Options for docs_feedback_sphinxext --------------------------------------

# NOTE: Must be one of 'attention', 'caution', 'danger', 'error', 'hint',
# NOTE: 'important', 'note', 'tip', 'warning' or 'admonition'.
docs_feedback_admonition_type = 'important'
docs_feedback_big_doc_lines = 50  # bigger docs will have a banner on top
docs_feedback_email = 'Docs UX Team <docs-feedback@pypa.io>'
docs_feedback_excluded_documents = {  # these won't have any banners
    'news', 'reference/index',
}
docs_feedback_questions_list = (
    'What problem were you trying to solve when you came to this page?',
    'What content was useful?',
    'What content was not useful?',
)

# -- Options for towncrier_draft extension -----------------------------------

towncrier_draft_autoversion_mode = 'draft'  # or: 'sphinx-release', 'sphinx-version'
towncrier_draft_include_empty = False
towncrier_draft_working_directory = pathlib.Path(docs_dir).parent
# Not yet supported: towncrier_draft_config_path = 'pyproject.toml'  # relative to cwd
