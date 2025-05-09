import numpy as np

class GraphWithFaces:
    """
    GraphWithFaces represents a graph defined by its faces and holes. When the object
    is not a cell complex a set of edges is also needed.
    It supports computing both the primal and dual graphs, and allows sorting edges
    based on a scalar signal defined on vertices.

    Parameters:
    -----------
    F : list of lists of int
        Faces of the graph, each face is a list of vertex indices starting at 0.
    H : list of lists of int
        Holes in the graph, each hole is a list of vertex indices.
    E : np.ndarray of shape (n, 2), optional
        List of edges. If not provided, edges are inferred from F and H.
    signal : np.ndarray of shape (n_vertices,)
        Scalar values defined at vertices, used for edge sorting.
    compute : {'normal', 'dual', 'both'}
        Whether to compute only the primal graph, only the dual, or both.
    is_triangulated : bool
        If True, the graph is assumed to be a triangular mesh.
    """
    
    def __init__(self, F=None, H=None, E=None, signal=None, compute="normal", is_triangulated=False):
        self.F = F 
        self.H = H
        self.E = E
        self.signal = signal
        self.E_signal = None
        self.dualE = set()
        self.dualE_signal = None
        self.compute = compute
        self.is_triangulated = is_triangulated
        self.vertex_count = signal.shape[0]
        self._extend_signal()

        if self.E is None:
            self.E = set()
            self._determine_E()
        elif compute != "normal":
            self._compute_dual()
        else:
            self._finalize_edges()

    def __str__(self):
        """
        Returns a human-readable string representation of the object.
        """
        return (
            f"Faces: {self.F},\n"
            f"Holes: {self.H},\n"
            f"Signal: {self.signal},\n"
            f"Edges: {self.E},\n"
            f"Edge signal: {self.E_signal}\n"
            f"Dual Edges: {self.dualE},\n"
            f"Dual Edge signal: {self.dualE_signal}"
        )

    def __repr__(self):
        """
        Returns a developer-friendly representation of the object.
        """
        return f"GraphWithFaces(F={self.F}, H={self.H}, signal={self.signal}, E={self.E}, E_signal={self.E_signal})"

    def _extend_signal(self):
        """
        Extends the input signal by adding values for face and hole centers,
        as needed for dual vertex construction. Faces get max signal value
        of their vertices, holes get +inf.
        """
        if self.is_triangulated:
            self.signal = np.concatenate((self.signal, np.full(len(self.H), np.inf)))
        else:
            maxF = [max([self.signal[v] for v in f]) for f in self.F]
            self.signal = np.concatenate((self.signal, np.array(maxF), np.full(len(self.H), np.inf)))

    def _finalize_edges(self):
        """
        Sorts edges and dual edges by signal values (maximum value along edge).
        Also stores edge signal arrays used for downstream filtering.
        """
        def sort_edges(edge_set, negate=False):
            edges = np.array(list(edge_set), dtype=np.uint32)
            values = self.signal if not negate else -self.signal
            edge_signals = np.maximum(values[edges[:, 0]], values[edges[:, 1]])
            sorted_idx = np.argsort(edge_signals)
            return edges[sorted_idx], edge_signals[sorted_idx]

        if self.compute in {"normal", "both"}:
            self.E, self.E_signal = sort_edges(self.E)

        if self.compute in {"dual", "both"}:
            self.dualE, self.dualE_signal = sort_edges(self.dualE, negate=True)

    def _determine_E(self):
        """
        Infers edge set from the boundary of faces and holes.
        Also builds dual edges if needed and sorts all edges.
        """
        for face in self.F:
            self._add_edges(face)
        for i in range(len(self.H) - (self.compute == "normal")):
            self._add_edges(self.H[i], ishole=True)
        self._finalize_edges()

    def _add_edges(self, sequence, *, ishole=False):
        """
        Adds primal and/or dual edges from a sequence of vertices (face or hole).

        Parameters:
        -----------
        sequence : list of int
            A closed walk representing a face or hole.
        ishole : bool
            Whether the sequence represents a hole (affects triangulated treatment).
        """
        n = len(sequence)
        need_dual = self.compute in {"dual", "both"}
        need_primal = self.compute in {"normal", "both"} or (self.is_triangulated and not ishole)

        add_edge = lambda u, v, dest: dest.add((min(u, v), max(u, v)))

        if need_dual:
            v3 = self.vertex_count
            self.vertex_count += 1

        for i in range(n):
            v1, v2 = sequence[i], sequence[(i + 1) % n]
            if need_primal:
                add_edge(v1, v2, self.E)
            if need_dual:
                self.dualE.add((v1, v3))
                add_edge(v1, v2, self.dualE)

        if self.is_triangulated and need_dual and ishole:
            v3 = self.vertex_count
            self.vertex_count += 1
            for v in sequence:
                self.dualE.add((v, v3))

    def _compute_dual(self):
        """
        Computes the dual graph: each face/hole becomes a new vertex,
        connected to all vertices in the corresponding face/hole.
        """
        def connect_dual_vertex(polygon):
            v_new = self.vertex_count
            self.vertex_count += 1
            for v in polygon:
                self.dualE.add((v, v_new))

        if not self.is_triangulated:
            for face in self.F:
                connect_dual_vertex(face)
        for hole in self.H:
            connect_dual_vertex(hole)

        self.dualE = np.vstack((np.array(list(self.E), dtype=np.uint32), np.array(list(self.dualE), dtype=np.uint32)))
        self._finalize_edges()
