from .filter_graph import TopologicalFilterGraph
from .filter_image import TopologicalFilterImage
from .gwf import GraphWithFaces
from .persistence_filter import Filtered, PersistenceFilter
from .tools import tools

__all__ = [
    "TopologicalFilterImage",
    "TopologicalFilterGraph",
    "GraphWithFaces",
    "PersistenceFilter",
    "Filtered",
    "tools",
]

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # For Python < 3.8, you can use the importlib-metadata package
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("topapprox")
except PackageNotFoundError:
    __version__ = "unknown"
    
