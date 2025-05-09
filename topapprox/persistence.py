""" Static functions for computing persistence diagrams
and related quantities
"""

from .filter_graph import TopologicalFilterGraph

def get_PD_gwf(F, H, signal, E=None):
    """ Computes the 0 and 1 dimensional persistence diagram
    of a graph with faces. The graph with faces is determined
    byt the set of faces `F`, set of holes `H`, and optional
    set of edges `E`. A function over the vertices is given as
    `signal`, for which this function considers the sub-level
    filtration.
    """

    # 0-homology
    tfg0 = TopologicalFilterGraph()
    tfg0.compute_gwf(F=F, H=H, E=E, signal=signal)
    _ = tfg0.low_pers_filter(epsilon=0)
    pd0 = tfg0.bht.get_persistence(reduced=False)
    if len(pd0.shape) > 1:
        pd0 = pd0[:, :2]

    # 1-homology
    tfg1 = TopologicalFilterGraph(dual=True)
    tfg1.compute_gwf(F=F, H=H, E=E, signal=signal)
    _ = tfg1.low_pers_filter(epsilon=0)
    pd1 = tfg1.bht.get_persistence()
    if len(pd1.shape) > 1:
        pd1 = pd1[:, :2]
    else:
        return [pd0]

    return [pd0, pd1]