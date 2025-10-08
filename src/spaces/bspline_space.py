import numpy as np
from tabulate import tabulate
from src.containers import Knotvector
from src.nurbs import NonUniformBSpline
from src.spaces import FunctionSpace


def _funcname(k, d):
    """
    Generate LaTeX-formatted function names for B-spline basis functions.

    Parameters
    ----------
    k : int
        Index of the B-spline basis function.
    d : int
        Derivative order (0 for basis function, >0 for derivatives).

    Returns
    -------
    str
        LaTeX-formatted string: "B_k" for basis functions, "B_k^(d)" for derivatives.

    Examples
    --------
    >>> _funcname(2, 0)
    '$B_{2}$'
    >>> _funcname(2, 1)
    '$B_{2}^{(1)}$'
    """
    text = r"$" + f"B" + r"_{" + f"{k}" + r"}$"
    if d > 0:
        text += r"${}^{(" + f"{d}" + r")}$"
    return text


class BSplineSpace(FunctionSpace):
    """
    A function space composed of B-spline basis functions.

    This class represents a complete B-spline function space defined by a knot vector.
    It automatically generates all valid B-spline basis functions for the given knot
    vector and provides operations for differentiation and space management. The class
    implements caching to avoid redundant computations.

    Parameters
    ----------
    t : Knotvector
        The knot vector defining the B-spline parameter space.
    derivative : int, optional
        The derivative order of the function space (default: 0).
    name : str, optional
        Name identifier for the function space (default: "BSpline").

    Attributes
    ----------
    t : Knotvector
        The knot vector used for definition.
    degree : int
        Degree of the B-spline basis functions.
    valid_function_keys : ndarray
        Array of valid indices for B-spline basis functions.
    nr_support_intervals : int
        Number of intervals in the support of each basis function.
    nr_support_knots : int
        Number of knots in the support of each basis function.
    derivative : int
        Derivative order of the function space.
    name : str
        Name identifier for the function space.
    registry : dict
        Class-level cache for storing computed B-spline spaces.
    key : tuple
        Unique identifier (t.key, derivative) for caching.

    Examples
    --------
    >>> # Create a quadratic B-spline space
    >>> t = Knotvector(degree=2, domain=(0, 1))
    >>> V = BSplineSpace(t, name="Quadratic_BSpline_Space")
    >>> print(f"Number of basis functions: {V.nr_items}")
    >>> print(f"Valid function keys: {V.valid_function_keys}")
    >>>
    >>> # Create derivative space
    >>> V_prime = V.diff(1)
    >>> print(f"Derivative space order: {V_prime.derivative}")

    Notes
    -----
    - Automatically generates all valid B-spline basis functions for the knot vector
    - Uses LaTeX formatting for function names for better visualization
    - Implements efficient caching to avoid redundant computations
    - Supports derivative spaces for variational formulations
    - Essential for isogeometric analysis and finite element methods
    """
    registry = dict()

    @classmethod
    def list(cls):
        """
        List all cached BSplineSpace instances in a formatted table.

        Returns
        -------
        str
            Formatted table showing space names, knot vectors, derivative orders,
            and function information for all cached B-spline spaces.
        """
        text = cls.__name__ + "\n" +\
               tabulate([(v.name, k[0].name, k[1], str(v)) for (k, v) in cls.registry.items()],
                        headers=("Name", "Knotvector", "Derivative", "Functions"), tablefmt='orgtbl')
        return text

    @property
    def key(self):
        """
        Unique identifier tuple for caching.

        Returns
        -------
        tuple
            Tuple containing (t.key, derivative) for unique identification
            and caching purposes.
        """
        return tuple(np.concatenate([self.t.key, self.derivative], axis=None))

    def __init__(self, t, derivative=0, name="BSpline"):
        """
        Initialize a B-spline function space.

        Parameters
        ----------
        t : Knotvector
            Knot vector defining the B-spline parameter space.
        derivative : int, optional
            Initial derivative order of the function space (default: 0).
        name : str, optional
            Name identifier for the function space (default: "BSpline").

        Notes
        -----
        - First checks registry cache before computation
        - Automatically generates all valid B-spline basis functions
        - Uses LaTeX formatting for function names
        - Stores computed spaces in registry for reuse
        - Sets up support properties for numerical integration
        """
        # Retrieve from registry
        self.derivative = derivative
        self.t = t

        if self.key in BSplineSpace.registry:
            mycopy: BSplineSpace = BSplineSpace.retrieve(self.key)
            self.degree = mycopy.degree
            self.name = mycopy.name
            self.keys = mycopy.keys
            self.items = mycopy.items
            self.valid_function_keys = mycopy.valid_function_keys
            self.nr_support_intervals = mycopy.nr_support_intervals
            self.nr_support_knots = mycopy.nr_support_knots
            return

        degree = t.degree
        valid_function_keys = np.arange(t.kmin, t.kmax + 1)
        items = [NonUniformBSpline(t, degree + 1, k, name=_funcname(k, derivative)) for k in valid_function_keys]
        super().__init__(keys=valid_function_keys, functions=items, derivative=derivative, name=name)

        self.valid_function_keys = np.arange(t.kmin, t.kmax + 1)
        self.degree = degree
        self.nr_support_intervals = self.degree
        self.nr_support_knots = self.degree + 1

        BSplineSpace.store(key=self.key, item=self)

    def diff(self, n: int = 1, copy=True, name=None):
        """
        Compute the n-th derivative of the B-spline function space.

        Creates a new function space containing the n-th derivatives of all
        B-spline basis functions. Updates function names with derivative notation
        and manages caching for efficient reuse.

        Parameters
        ----------
        n : int, optional
            Order of derivative to apply (default: 1).
        copy : bool, optional
            If True, returns a derivative copy; if False, differentiates in-place.
        name : str, optional
            New name for the derivative function space.

        Returns
        -------
        BSplineSpace
            B-spline function space containing the n-th derivatives of all
            basis functions. Returns copy if copy=True.

        Examples
        --------
        >>> # Create B-spline space and its first derivative
        >>> V = BSplineSpace(t)
        >>> V_prime = V.diff(1, name="BSpline_Derivative")
        >>> print(f"Derivative order: {V_prime.derivative}")
        1
        >>> print(f"Function names: {[f.name for f in V_prime.items]}")
        ['$B_{0}^{(1)}$', '$B_{1}^{(1)}$', ...]

        Notes
        -----
        - Checks registry cache before computing derivatives
        - Updates function names with LaTeX derivative notation
        - Automatically caches derivative spaces for reuse
        - Maintains the same organizational structure as the original space
        - Essential for problems requiring derivative basis functions in weak forms
        """
        new_key = self.t, self.derivative+n
        if copy:
            # Retrieval works with deepcopies, so it should be used if copy is False.
            if new_key in BSplineSpace.registry:
                return BSplineSpace.registry[new_key]

            dv = super().diff(n,copy)
            for k in dv.keys:
                dv[k].name = _funcname(k, n)

            if name is not None: dv.name = name

            BSplineSpace.store(key=new_key, item=dv)
            return dv
        else:
            super().diff(n, copy)
            if name is not None: self.name = name
            BSplineSpace.store(key=new_key, item=self)
            return self
