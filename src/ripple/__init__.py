"""Cryo-EM movie frame alignment and polishing.

RIPPLE: Realigning image patches and polishing local environment
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("RIPPLE")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"
