import numpy as np

NumericValues = (int, float, complex, np.int32, np.float64, np.complex128)
"""
A tuple of numeric types supported for mathematical operations.

This collection defines the standard numeric types that can be used in 
mathematical computations, including both Python built-in types and 
commonly used NumPy numeric types. It is typically used for type checking
in mathematical libraries and numerical computation frameworks.

Notes
-----
- This tuple is designed for use with `isinstance()` checks
- Covers both Python's abstract numeric hierarchy and specific NumPy dtypes
- Useful for validating inputs in numerical algorithms and mathematical expressions
- Can be extended with additional NumPy types if needed for specific applications
- Not all type are needed in current implementation (e.g. complex numbers), but are added for future extensions
"""