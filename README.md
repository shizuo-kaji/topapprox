# topapprox

A Python package for **persistent homology based filtering** of signals over graphs and images.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Authors**:
[Matias de Jong van Lier](https://sites.google.com/view/matiasvanlier), [Junyan Chu](https://sites.google.com/view/junyan-chu/), Sebastían Elías Graiff Zurita, [Shizuo Kaji](https://www.skaji.org/)
**Copyright © 2024**

---

## Caution

This package is in early development. APIs are subject to change in future versions.

---

## Overview

**topapprox** implements topological filtering methods based on persistent homology, designed for:

- 1D and 2D **images**
- Functions defined on the **nodes of a graph**

It enables signal denoising and structure-preserving smoothing using techniques inspired by topological data analysis (TDA), including the **Basin Hierarchy Tree (BHT)**.

---

## Documentation & Examples

- [Interactive Tutorial](notebook/Interactive_Tutorial_topapprox.ipynb)
- [Reproducible Examples from Our Paper](notebook/Reproducing_paper_examples.ipynb)
- [Original Paper on arXiv (2024)](https://arxiv.org/abs/2408.14109)

---

## Installation

```bash
pip install git+https://github.com/shizuo-kaji/topapprox@main
```

## Quick Start

```python
import numpy as np
import topapprox as ta
from topapprox.persistence import get_PD_gwf

# Image filtering (0-homology and 1-homology via dual=True)
img = np.array([[0, 5, 3], [5, 6, 4], [2, 5, 1]], dtype=float)
tfi = ta.TopologicalFilterImage(img, method="python")
filtered_h0 = tfi.low_pers_filter(1.5)

tfi_dual = ta.TopologicalFilterImage(img, dual=True, method="python")
filtered_h1 = tfi_dual.low_pers_filter(1.5)

# Graph-with-faces filtering
faces = [[0, 1, 2, 3]]
holes = [[0, 1, 2, 3]]
signal = np.array([0.0, 1.0, 0.5, 0.2])
tfg = ta.TopologicalFilterGraph(method="python")
tfg.compute_gwf(F=faces, H=holes, signal=signal)
filtered_graph = tfg.low_pers_filter(0.5)
pd0, pd1 = tfg.get_diagram()

# Unified wrapper (alternating filtering orders, caching)
pf = ta.PersistenceFilter().load_signal(img)
filtered_01 = pf.low_pers_filter(1.5, iteration_order="01", method="python")

# Convenience persistence helper
pd0_fn, pd1_fn = get_PD_gwf(faces, holes, signal, method="python")
```

## Citation

If you use this package in your work, please cite:

> Matias de Jong van Lier, Sebastían Elías Graiff Zurita, Shizuo Kaji.
> **Topological filtering of a signal over a network** (2024).
> [arXiv:2408.14109](https://arxiv.org/abs/2408.14109)

```bibtex
@article{vanlier2024topological,
  title={Topological filtering of a signal over a network},
  author={de Jong van Lier, Matias and Graiff Zurita, Sebastían Elías and Kaji, Shizuo},
  journal={arXiv preprint arXiv:2408.14109},
  year={2024}
}
