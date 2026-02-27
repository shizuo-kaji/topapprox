""" Static functions for computing persistence diagrams
and related quantities
"""

import numpy as np

from .filter_graph import TopologicalFilterGraph

def get_PD_gwf(F, H, signal, E=None, *, method="cpp", bht_method="python", is_triangulated=False):
    """ Computes the 0 and 1 dimensional persistence diagram
    of a graph with faces. The graph with faces is determined
    byt the set of faces `F`, set of holes `H`, and optional
    set of edges `E`. A function over the vertices is given as
    `signal`, for which this function considers the sub-level
    filtration.
    """

    tfg = TopologicalFilterGraph(
        method=method,
        bht_method=bht_method,
        dual=False,
        is_triangulated=is_triangulated,
    )
    tfg.compute_gwf(F=F, H=H, E=E, signal=np.asarray(signal))
    pd0, pd1 = tfg.get_diagram()
    return [np.asarray(pd0), np.asarray(pd1)]
