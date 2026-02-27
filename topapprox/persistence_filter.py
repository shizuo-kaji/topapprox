"""Unified persistence filtering wrapper.

This module combines image and graph filtering interfaces in a single class.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .filter_graph import TopologicalFilterGraph
from .filter_image import TopologicalFilterImage


@dataclass(frozen=True)
class Filtered:
    """Container for a cached filtered signal."""

    signal: np.ndarray
    iteration_order: Tuple[str, ...]


class PersistenceFilter:
    """Perform low persistence filtering on arrays and graph-with-faces signals.

    Supported inputs:
    1. ``np.ndarray`` (1D/2D/3D image signal)
    2. ``[faces, holes, signal]`` or ``[faces, holes, signal, edges]`` for graphs
       where ``faces`` and ``holes`` are lists and ``signal`` is an ``np.ndarray``.
    """

    def __init__(self) -> None:
        self.type = None
        self.signal = None
        self.dim = None
        self.faces = None
        self.holes = None
        self.edges = None
        self._graph_n_vertices = None

        # Public history (kept for backwards compatibility).
        self.bht = []
        self.filtered = []
        self.filtered_type = []  # each type is a tuple[str, ...]

        # Internal caches.
        self._filtered_cache: Dict[Tuple[str, ...], np.ndarray] = {}
        self._bht_cache: Dict[Tuple[str, ...], object] = {}
        self._last_key: Tuple[str, ...] = ()

    def _reset_cache(self) -> None:
        self.bht = []
        self.filtered = []
        self.filtered_type = []
        self._bht_cache = {}
        self._filtered_cache = {}
        self._last_key = ()
        if self.signal is not None:
            self._filtered_cache[()] = np.array(self.signal, copy=True)

    def load_signal(self, signal):
        """Load a signal and reset all cached filtering results."""
        if isinstance(signal, np.ndarray):
            self.signal = np.array(signal, copy=True)
            self.type = "array"
            self.dim = self.signal.ndim
            self.faces = None
            self.holes = None
            self.edges = None
            self._graph_n_vertices = None
            self._reset_cache()
            return self

        if isinstance(signal, (list, tuple)):
            if len(signal) not in (3, 4):
                raise TypeError(
                    "Graph input should be [faces, holes, signal] or "
                    "[faces, holes, signal, edges]."
                )
            faces, holes, values = signal[:3]
            edges = signal[3] if len(signal) == 4 else None
            if not isinstance(faces, list) or not isinstance(holes, list):
                raise TypeError("faces and holes must be lists.")
            if not isinstance(values, np.ndarray):
                raise TypeError("graph signal values must be provided as np.ndarray.")

            self.faces = faces
            self.holes = holes
            self.edges = edges
            self.signal = np.array(values, copy=True).ravel()
            self.type = "graph"
            self.dim = 1
            self._graph_n_vertices = self.signal.shape[0]
            self._reset_cache()
            return self

        raise TypeError(
            "signal should be a numpy array, or [faces, holes, signal] "
            "(optionally with edges as a 4th element)."
        )

    @staticmethod
    def _normalize_iteration_order(iteration_order) -> Tuple[str, ...]:
        if isinstance(iteration_order, str):
            items = list(iteration_order)
        else:
            items = [str(x) for x in iteration_order]
        if any(x not in {"0", "1"} for x in items):
            raise ValueError("iteration_order should contain only '0' and '1'.")
        return tuple(items)

    @staticmethod
    def _epsilon_for_step(epsilon, step: str) -> float:
        if np.isscalar(epsilon):
            return float(epsilon)
        if isinstance(epsilon, dict):
            if step in epsilon:
                return float(epsilon[step])
            key_int = int(step)
            if key_int in epsilon:
                return float(epsilon[key_int])
            raise KeyError(f"Missing epsilon for homology {step}.")
        if isinstance(epsilon, Sequence):
            if len(epsilon) != 2:
                raise ValueError(
                    "When epsilon is a sequence, it should have length 2 "
                    "(for homology 0 and 1)."
                )
            return float(epsilon[int(step)])
        raise TypeError(
            "epsilon should be a scalar, a dict keyed by 0/1 (or '0'/'1'), "
            "or a length-2 sequence."
        )

    def _build_filter(
        self,
        current_signal: np.ndarray,
        *,
        dual: bool,
        method: str,
        bht_method: str,
        recursive: bool,
        iter_vertex: bool,
        is_triangulated: bool,
    ):
        if self.type == "array":
            return TopologicalFilterImage(
                np.array(current_signal, copy=True),
                method=method,
                bht_method=bht_method,
                dual=dual,
                recursive=recursive,
                iter_vertex=iter_vertex,
            )

        if self.type == "graph":
            graph_filter = TopologicalFilterGraph(
                method=method,
                bht_method=bht_method,
                dual=dual,
                recursive=recursive,
                is_triangulated=is_triangulated,
            )
            graph_filter.compute_gwf(
                self.faces,
                self.holes,
                np.array(current_signal, copy=True).ravel(),
                E=self.edges,
            )
            return graph_filter

        raise RuntimeError("No signal has been loaded. Call load_signal first.")

    def low_pers_filter(
        self,
        epsilon,
        *,
        iteration_order="01",
        method="cpp",
        bht_method="python",
        recursive=True,
        iter_vertex=True,
        is_triangulated=False,
        size_range=None,
        return_sequence=False,
    ):
        """Apply low persistence filtering in the requested homology order.

        Parameters
        ----------
        epsilon:
            Scalar threshold used for every step, or per-homology values via
            ``{0: e0, 1: e1}``, ``{"0": e0, "1": e1}``, or ``[e0, e1]``.
        iteration_order:
            String/iterable made of ``0`` and ``1`` (e.g. ``"0"``, ``"01"``,
            ``"101"``). ``0`` means primal filtering, ``1`` means dual filtering.
        return_sequence:
            If ``True``, also returns intermediate cached snapshots.
        """
        if self.signal is None:
            raise RuntimeError("No signal has been loaded. Use load_signal(...) first.")

        order = self._normalize_iteration_order(iteration_order)
        if len(order) == 0:
            self._last_key = ()
            out = np.array(self.signal, copy=True)
            return (out, []) if return_sequence else out

        # Reuse the longest available prefix.
        prefix_len = len(order)
        while prefix_len >= 0 and order[:prefix_len] not in self._filtered_cache:
            prefix_len -= 1
        if prefix_len < 0:
            prefix_len = 0

        current_key = order[:prefix_len]
        current = np.array(self._filtered_cache[current_key], copy=True)
        sequence = []

        for idx in range(prefix_len, len(order)):
            step = order[idx]
            dual = step == "1"
            step_epsilon = self._epsilon_for_step(epsilon, step)
            filter_obj = self._build_filter(
                current,
                dual=dual,
                method=method,
                bht_method=bht_method,
                recursive=recursive,
                iter_vertex=iter_vertex,
                is_triangulated=is_triangulated,
            )
            current = np.array(
                filter_obj.low_pers_filter(step_epsilon, size_range=size_range),
                copy=True,
            )
            if self.type == "graph" and self._graph_n_vertices is not None:
                current = current.ravel()[: self._graph_n_vertices].copy()
            key = order[: idx + 1]
            self._filtered_cache[key] = np.array(current, copy=True)
            self._bht_cache[key] = filter_obj.bht

            if key not in self.filtered_type:
                self.filtered_type.append(key)
                self.filtered.append(np.array(current, copy=True))
                self.bht.append(filter_obj.bht)

            sequence.append(Filtered(signal=np.array(current, copy=True), iteration_order=key))

        self._last_key = order

        if return_sequence:
            return np.array(self._filtered_cache[order], copy=True), sequence
        return np.array(self._filtered_cache[order], copy=True)

    def get_filtered(self, iteration_order: Optional[Iterable[str]] = None) -> np.ndarray:
        """Return a cached filtered signal."""
        if self.signal is None:
            raise RuntimeError("No signal has been loaded. Use load_signal(...) first.")
        key = self._last_key if iteration_order is None else self._normalize_iteration_order(iteration_order)
        if key not in self._filtered_cache:
            raise KeyError(
                f"No cached result for iteration_order={''.join(key)}. "
                "Run low_pers_filter(...) first."
            )
        return np.array(self._filtered_cache[key], copy=True)

    def get_BHT(self, iteration_order: Optional[Iterable[str]] = None):
        """Return the BHT of a cached filtering run."""
        key = self._last_key if iteration_order is None else self._normalize_iteration_order(iteration_order)
        if len(key) == 0:
            raise KeyError("No BHT exists for the empty iteration order.")
        if key not in self._bht_cache:
            raise KeyError(
                f"No cached BHT for iteration_order={''.join(key)}. "
                "Run low_pers_filter(...) first."
            )
        return self._bht_cache[key]
