"""Package for image analysis."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("iob-ia")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Loïc Sauteur"
__email__ = "loic.sauteur@unibas.ch"
