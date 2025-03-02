"""ALineMol: Package for evaluating out-of-distribution performance for molecules."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("alinemol")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"

# Import and expose key functionality
from .models import *  # Import your models
from .preprocessing import *  # Import preprocessing tools
from .utils import *  # Import utility functions