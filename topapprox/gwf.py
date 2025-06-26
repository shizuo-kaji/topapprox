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
        List of edges. If not provided, edges are inferred from F and H, which is
        only possible when the graph with faces is a cell complex, meaning that
        each face should be homeomorphic to an open disk and its boundary
        homeomorphic to a circle. In any other case the set of edges should be provided.
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

        self.n_nodes = len(set([x for edge in self.E for x in edge]))

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





    ##############################
    ##############################
    # For Drawing ################
    ##############################
    ##############################

    def draw(
        self,
        signal=None,
        pos=None,
        cmap='viridis',
        figsize=(6, 6),
        face_alpha=0.4,
        node_edgecolors='black',
        colorbar=True,
        edge_width=2.5,
        vmin=None,
        vmax=None,
        edge_boundary_color=None,
        linewidth=1.5,
        threshold=None,
        ax = None,
        gray_faces = False          
    ):
        """
        Plot a signal on the vertices, with edge and face colours given by the
        *maximum* signal on their vertices.
        If `threshold` is not None, elements with value >= threshold are rendered
        fully transparent (alpha = 0) but are still created so that aspect, ticks,
        etc. stay unchanged.

        Parameters
        ----------
        threshold : float or None, optional
            Draw only elements whose value is < threshold.  Others are drawn
            transparent.  Default None (plot everything normally).
        <---- rest of the original docstring, unchanged ---->
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import networkx as nx

        if signal is None:
            signal = self.signal[:self.n_nodes]  # use only vertex signals

        if ax is None:                       
            fig, ax = plt.subplots(figsize=figsize)
        else:                             
            fig = ax.figure                 
            ax.clear()

        if len(signal) != self.n_nodes:
            raise ValueError("Signal length must match number of nodes.")

        # ---------- graph / layouts ----------
        G = nx.Graph()
        G.add_edges_from([tuple(e) for e in self.E])
        if pos is None:
            pos = nx.spring_layout(G, seed=42)

        # ---------- per-element signals ----------
        edge_signal = np.array([max(signal[u], signal[v]) for u, v in self.E])
        face_signal = np.array([max(signal[v] for v in face) for face in self.F])
        hole_signal = np.array([max(signal[v] for v in hole) for hole in self.H])

        if vmin is None:
            vmin = min(signal.min(), edge_signal.min(), face_signal.min())
        if vmax is None:
            vmax = max(signal.max(), edge_signal.max(), face_signal.max())

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap_func = plt.cm.get_cmap(cmap)

        # helper: transparent version of a colour  -------------------------------
        def transparent(col):
            r, g, b, _ = mcolors.to_rgba(col)
            return (r, g, b, 0.0)


        edge_color_face = "black" if threshold is None else 'black'
        # ---------- holes ----------
        if self.H is not None:
            for i, hole in enumerate(self.H):
                polygon = np.array([pos[v] for v in hole])
                if threshold is not None and hole_signal[i] >= threshold:
                    edge_color_face = "none"
                    alpha = 0.0                    # hide completely
                else:
                    edge_color_face = "black"
                    alpha = face_alpha
                ax.fill(*zip(*polygon), color='none', alpha=alpha,
                        edgecolor=edge_color_face, linewidth=linewidth)

        
        # ---------- faces ----------
        for i, face in enumerate(self.F):
            polygon = np.array([pos[v] for v in face])
            col = cmap_func(norm(face_signal[i]))
            if threshold is not None and face_signal[i] >= threshold:
                col = transparent(col)
                alpha = 0.0                    # hide completely
                edge_color_face = "none"
            else:
                alpha = face_alpha
                edge_color_face = "black"
            if gray_faces:
                col = 'lightgray'
            ax.fill(*zip(*polygon), color=col, alpha=alpha,
                    edgecolor=edge_color_face, linewidth=linewidth)

        # ---------- edges ----------
        for i, (u, v) in enumerate(self.E):
            if edge_boundary_color is not None:
                col = edge_boundary_color
            else:
                col = cmap_func(norm(edge_signal[i]))

            if threshold is not None and edge_signal[i] >= threshold:
                col = transparent(col)

            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=col, linewidth=edge_width, solid_capstyle='round')

        # ---------- nodes ----------
        node_face_colors = []
        node_edge_colors = []

        for val in signal:
            base_rgba = cmap_func(norm(val))

            if threshold is not None and val >= threshold:
                # vertex is “masked”: completely transparent face, outline transparent
                node_face_colors.append(transparent(base_rgba))
                node_edge_colors.append(transparent('black'))     # or transparent(base_rgba) if you prefer
            else:
                # vertex is shown: normal face colour, but *transparent* outline
                node_face_colors.append(base_rgba)
                node_edge_colors.append(node_edgecolors)  # invisible contour

        node_xy = np.array([pos[i] for i in range(self.n_nodes)])
        ax.scatter(node_xy[:, 0], node_xy[:, 1],
                c=node_face_colors,
                edgecolors=node_edge_colors,
                s=200, vmin=vmin, vmax=vmax, zorder=10)

        # ---------- colour-bar ----------
        if colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(signal)
            plt.colorbar(sm, ax=ax, orientation='vertical',
                        fraction=0.03, pad=0.02, shrink=0.6)

        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig, ax