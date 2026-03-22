try:
    import importlib.metadata as _im
except ImportError:
    import importlib_metadata as _im
try:
    __version__ = _im.version("yxsplot")
except ImportError:
    __version__ = "0+unknown"

from .core import plot, show_figure

__all__ = ["plot", "show_figure"]
