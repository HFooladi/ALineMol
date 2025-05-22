"""Version information."""

try:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version("ALineMol")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "0.0.0.dev0"
except ImportError:
    # Python < 3.8
    __version__ = "0.0.0.dev0"  # package is not installed
