# FAST-B: Fast Analytic B-spline Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JOSS Paper](https://joss.theoj.org/papers/10.21105/joss.xxxxx/status.svg)
](https://doi.org/10.21105/joss.xxxxx)

A high-performance Python library for generating analytic B-spline expressions 
using NumPy instead of symbolic computation. FAST-B provides mathematically 
equivalent results to symbolic approaches but with significantly improved 
performance through intelligent caching and efficient numerical computation.

## Key Features

- **⚡ High Performance**: 10-1000× faster than symbolic computation through memoization
- **🎯 Mathematical Precision**: Faithful implementation of Cox-de Boor algorithm
- **🔧 Flexible Boundaries**: Support for mixed open/closed boundary conditions
- **📈 IGA Ready**: Direct application to Isogeometric Analysis
- **🎨 LaTeX Output**: Beautiful mathematical representation of functions
- **💾 Efficient Caching**: Automatic memoization of computed expressions

## Installation

```bash
git clone https://github.com/tctheodsiou/fast-b.git
cd fast-b
pip install -e .
```

## Quick start
```python
import numpy as np
import matplotlib.pylab as plt
from src.nurbs.bsplines import CardinalBSpline

# Create a quadratic B-spline
B = CardinalBSpline(m=2)

# Evaluate at multiple points
x = np.linspace(0, 3, 1000)
y = B(x)

# Compute derivatives
B_prime = B.diff(1)  # First derivative

plt.figure()
plt.plot(x, B(x), label="B(x)")
plt.plot(x, B_prime(x), label="dB/dx")
plt.legend()
plt.show()

```

## Architecture
FAST-B is built on a hierarchical architecture:
```
src/
├── bsplines/         # B-spline implementations
├── containers/       # Knot vectors and data structures
├── nurbs/            # NURBS support
├── piecewise/        # Piecewise function framework
├── spaces/           # Function spaces
└── system/           # Caching and core utilities
```

## Core components
- **Polynomial Foundation:** Enhanced numpy.poly1d with algebraic operations
- **Rational Functions:** Support for NURBS and rational B-splines 
- **Piecewise Framework:** Domain-aware function management
- **Caching System:** Intelligent memoization for performance
- **B-spline Engine:** Cox-de Boor algorithm implementation

## Research Applications
FAST-B has been successfully used in:
- Isogeometric Analysis: Efficient basis function generation for IGA
- Wavelet Development: Creation of derivative-orthogonal non-uniform B-spline 
  wavelets
- Computational Mathematics: Research in numerical methods and approximation 
  theory
- Engineering Applications: finite element analysis, and geometric modeling

## License
This project is licensed under the MIT License.
