"""
Graph with faces class

This class is used to represent a graph with faces.
By passing a signal value the edges are automatically sorted accordingly.
One can compute the dual of the graph with faces by setting `compute` to "dual",
if it is set to "both", then both the normal and dual versions will be computed
"""


import numpy as np

class GraphWithFaces:
    def __init__(self, F=None, H=None, E=None, signal=None, compute="normal", is_triangulated=False):
        """
        Initializes the graph with faces. Faces (F) and holes (H) should be
        lists of lists, where each inner list represents a sequence of vertices
        that form a face or hole.

        :param F: List of faces, each face is a list of vertices.
        :param H: List of holes, each hole is a list of vertices.
        :param E: (optional) numpy array  of shape (n,2), with n being the number of edges,
        each entry should be one edge.
            If the graph embedding is such that each face is homeomorphic to an open disk 
            and its boundary is homemorphic to a circle, then there is no need to pass a list
            of edges, as long as the faces an holes are given in an appropriate order (each face
            or hole should be written for example as face = [1,2,3,4], meaning that 12, 23, 34 and 41
            are the edges forming that face/hole).
        :param signal: A NumPy array of shape (n_vertices,) representing the function values.
        :param compute: A string, can be either "normal", "dual" or "both"
        :param ismesh: bool, if True a special case is considered for mesh (all faces are triangles). Holes can be anything.
        """
        self.F = F if F is not None else []
        self.H = H if H is not None else []
        self.E = E
        self.signal = signal
        self.E_signal = None
        self.compute = compute
        self.vertex_count = signal.shape[0]
        self.is_triangulated = is_triangulated
        # Automatically determine edges from F and H if provided
        self.dualE = set()
        self.dualE_signal = None
        if is_triangulated:
            self.signal = np.concatenate((signal, np.full(len(H), np.inf)))
        else:
            self.signal = np.concatenate((signal, np.array([max([signal[v] for v in f]) for f in F]), np.full(len(H), np.inf)))
        if self.E is None:
            self.E = set()
            self._determine_E()
        elif compute != "normal":
            self._compute_dual()
        else:
            edges = np.array(list(self.E), dtype=np.uint32)
            # Compute the max signal value for each edge
            edge_signals = np.maximum(self.signal[edges[:, 0]], self.signal[edges[:, 1]])
            # Sort edges by signal values
            sorted_indices = np.argsort(edge_signals)
            self.E_signal = edge_signals[sorted_indices]
            self.E = edges[sorted_indices]


       

    def __str__(self):
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
        return f"GraphWithFaces(F={self.F}, H={self.H}, signal={self.signal}, E={self.E}, E_signal={self.E_signal})"

    def _determine_E(self):
        """
        Determines the edge set E based on the faces (F) and holes (H), and sorts it by signal values.
        """
        # Step 1: Add edges from faces and holes
        for face in self.F:
            self._add_edges(face)
        n = len(self.H) - 1 if self.compute == "normal" else len(self.H)
        for i in range(n): # We can skip the final face, since all its edges were already included
            self._add_edges(self.H[i], ishole=True)

        if self.compute == "normal" or self.compute == "both":
            # Convert set of edges to numpy array for efficient operations
            edges = np.array(list(self.E), dtype=np.uint32)
            # Compute the max signal value for each edge
            edge_signals = np.maximum(self.signal[edges[:, 0]], self.signal[edges[:, 1]])
            # Sort edges by signal values
            sorted_indices = np.argsort(edge_signals)
            self.E_signal = edge_signals[sorted_indices]
            self.E = edges[sorted_indices]

        if self.compute == "dual" or self.compute == "both":
            # Convert set of edges to numpy array for efficient operations
            dual_edges = np.array(list(self.dualE), dtype=np.uint32)
            # Compute the max signal value for each edge
            dual_edge_signals = np.maximum(-self.signal[dual_edges[:, 0]], -self.signal[dual_edges[:, 1]])
            # Sort edges by signal values
            dual_sorted_indices = np.argsort(dual_edge_signals)
            self.dualE_signal = dual_edge_signals[dual_sorted_indices]
            self.dualE = dual_edges[dual_sorted_indices]
        

    def _add_edges(self, sequence, *, ishole=False):
        """
        Adds edges for a given sequence of vertices, which can be a face or hole.
        :param sequence: A list of vertices.
        """
        n = len(sequence)
        if self.compute == "normal" or (self.is_triangulated and not ishole):
            for i in range(n):
                v1, v2 = sequence[i], sequence[(i + 1) % n]
                if v1 > v2:  # Manually check for edge direction to avoid sorting
                    v1, v2 = v2, v1
                self.E.add((v1, v2))  # Use set for fast duplicate detection
        elif self.compute == "dual":
            v3 = self.vertex_count # extra vertex
            self.vertex_count += 1
            for i in range(n):
                v1, v2 = sequence[i], sequence[(i + 1) % n]
                self.dualE.add((v1, v3))
                if v1 > v2:  # Manually check for edge direction to avoid sorting
                    v1, v2 = v2, v1
                self.dualE.add((v1, v2))
        elif self.compute == "both":
            v3 = self.vertex_count # extra vertex
            self.vertex_count += 1
            for i in range(n):
                v1, v2 = sequence[i], sequence[(i + 1) % n]
                self.dualE.add((v1, v3))
                if v1 > v2:  # Manually check for edge direction to avoid sorting
                    v1, v2 = v2, v1
                self.E.add((v1, v2))
                self.dualE.add((v1, v2))

        if self.is_triangulated and (self.compute != "normal") and ishole:
            v3 = self.vertex_count
            self.vertex_count += 1
            for v in sequence:
                self.dualE.add((v, v3))


    def _compute_dual(self):
        """ Computes the dual graph from the graph with faces
        Each face/hole becomes a new vertex which is attached by edges to all the 
        vertices in the face/hole. 
        """
        # Add edges from faces and holes
        if not self.is_triangulated:
            for face in self.F:
                v_new = self.vertex_count # extra vertex
                self.vertex_count += 1
                for v in face:
                    self.dualE.add((v,v_new))
        for hole in self.H:
                v_new = self.vertex_count # extra vertex
                self.vertex_count += 1
                for v in hole:
                    self.dualE.add((v,v_new))
        dual_edges = np.array(list(self.dualE), dtype=self.E.dtype)
        dual_edges = np.vstack((self.E, dual_edges))
        # Compute the max signal value for each edge
        dual_edge_signals = np.maximum(-self.signal[dual_edges[:, 0]], -self.signal[dual_edges[:, 1]])
        # Sort edges by signal values
        dual_sorted_indices = np.argsort(dual_edge_signals)
        self.dualE_signal = dual_edge_signals[dual_sorted_indices]
        self.dualE = dual_edges[dual_sorted_indices]