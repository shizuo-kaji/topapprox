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

- 🔗 [Interactive Tutorial](Interactive_Tutorial_topapprox.ipynb)
- 📝 [Reproducible Examples from Our Paper](https://github.com/mvlier/topapprox/blob/main/Paper%20examples/Reproducing_paper_examples.ipynb)
- 📄 [Original Paper on arXiv (2024)](https://arxiv.org/abs/2408.14109)

---

## Installation

```bash
pip install git+https://github.com/mvlier/topapprox@main
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

