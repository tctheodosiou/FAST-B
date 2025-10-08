import numpy as np
from tabulate import tabulate
from src.piecewise import Polynomial, Rational, Branch, Function
from src.containers import Knotvector


class CardinalBSpline(Function):
    """
    Cardinal B-spline basis functions with uniform knot spacing.

    This class implements cardinal B-spline basis functions defined on
    uniform knot vectors with integer knots. The basis functions are
    computed recursively using the Cox-de Boor algorithm and cached
    for efficient reuse.

    Parameters
    ----------
    m : int
        The degree of the B-spline basis function (m > 0).
    name : str, optional
        Optional name identifier for the basis function.

    Attributes
    ----------
    m : int
        Degree of the B-spline.
    derivative : int
        Derivative order (0 for basis function, >0 for derivatives).
    registry : dict
        Class-level cache for storing computed B-spline functions.
    key : tuple
        Unique identifier (m, derivative) for caching.

    Examples
    --------
    >>> # Create quadratic cardinal B-spline
    >>> B2 = CardinalBSpline(m=2)
    >>> B2(1.5)  # Evaluate at x=1.5
    0.5
    >>>
    >>> # First derivative
    >>> B2_prime = B2.diff(1)
    >>> B2_prime(1.5)
    -1.0

    Notes
    -----
    - Uses uniform integer knots: 0, 1, 2, ..., m
    - Implements recursive Cox-de Boor algorithm
    - Automatically caches computed functions for performance
    - Supports derivatives through automatic differentiation
    - Inherits from Function class for piecewise representation
    """
    registry = dict()

    @classmethod
    def list(cls):
        """
        List all cached CardinalBSpline instances in a formatted table.

        Returns
        -------
        str
            Formatted table showing degree, derivative order, and formulation
            of all cached B-spline functions.
        """
        text = cls.__name__ + "\n" + \
               tabulate([(f"CBS {k[0]}", f"{k[1]}", str(v)) for (k, v) in cls.registry.items()],
                        headers=("Degree", "Derivative", "Formulation"), tablefmt='orgtbl')
        return text

    @property
    def key(self):
        """Unique identifier tuple (m, derivative) for caching."""
        return self.m, self.derivative

    def __init__(self, m, name=None):
        """
        Initialize a Cardinal B-spline basis function.

        Parameters
        ----------
        m : int
            Degree of the B-spline (must be > 0).
        name : str, optional
            Optional name identifier.

        Raises
        ------
        AssertionError
            If m <= 0.

        Notes
        -----
        - First checks registry cache before computation
        - For m=1: creates constant basis function on [0,1]
        - For m>1: recursively builds from lower-degree B-splines
        - Automatically stores computed functions in registry
        """
        self.m = m
        self.derivative = 0

        # Retrive from registry.
        mycopy: CardinalBSpline = CardinalBSpline.retrieve(self.key)
        if mycopy is not None:
            self.build_from(mycopy)
            return

        # Otherwise, build from scratch, and store in registry.
        assert m > 0, "Degree must be larger than 0."

        if m == 1:
            B = Function(Branch(numerator=Polynomial(1), support=[0, m]))
        else:
            BS = CardinalBSpline(m - 1)

            F1 = BS
            C1 = Function(Branch(numerator=Polynomial([1, 0]), denominator=Polynomial(m - 1), support=(0, m)))
            G1 = C1 * F1

            F2 = BS.shift(-1)
            C2 = Function(Branch(numerator=Polynomial([-1, m]), denominator=Polynomial(m - 1), support=(0, m)))
            G2 = C2 * F2

            B = G1 + G2
            B.simplify()

        super().__init__(B.branches, name=name)
        CardinalBSpline.store(key=self.key, item=self)

    def diff(self, n: int = 1, copy=True, name=None):
        """
        Compute the n-th derivative of the B-spline basis function.

        Parameters
        ----------
        n : int, optional
            Order of derivative (default: 1).
        copy : bool, optional
            If True, returns a derivative copy; if False, differentiates in-place.
        name : str, optional
            Optional name for the derivative function.

        Returns
        -------
        CardinalBSpline
            The n-th derivative as a new CardinalBSpline object.

        Notes
        -----
        - Checks registry cache before computing derivatives
        - Automatically caches computed derivatives
        - Updates the derivative attribute accordingly
        """
        new_key = self.m, self.derivative + n

        if copy:
            # Retrieval works with deepcopies, so it should be used if copy is False.
            if new_key in CardinalBSpline.registry:
                return CardinalBSpline.registry[new_key]

            dv = super().diff(n, copy, name)
            CardinalBSpline.store(key=new_key, item=dv)
            return dv

        else:
            super().diff(n, copy, name)
            CardinalBSpline.store(key=new_key, item=self)
            return self


class NonUniformBSpline(Function):
    """
    Non-uniform B-spline basis functions for arbitrary knot vectors.

    This class implements B-spline basis functions defined on non-uniform
    knot vectors. The basis functions are computed recursively using the
    Cox-de Boor algorithm and support arbitrary knot distributions.

    Parameters
    ----------
    t : Knotvector
        The knot vector defining the B-spline parameter space.
    m : int
        The degree of the B-spline basis function (m > 0).
    k : int
        The index of the basis function within the knot vector.
    name : str, optional
        Optional name identifier for the basis function.

    Attributes
    ----------
    t : Knotvector
        The knot vector used for definition.
    m : int
        Degree of the B-spline.
    k : int
        Index of the basis function.
    derivative : int
        Derivative order (0 for basis function, >0 for derivatives).
    registry : dict
        Class-level cache for storing computed B-spline functions.
    key : tuple
        Unique identifier (t.key, m, k, derivative) for caching.

    Examples
    --------
    >>> # Create knot vector and B-spline
    >>> t = Knotvector(degree=2, domain=(0, 1))
    >>> N = NonUniformBSpline(t, m=2, k=0)
    >>> N(0.5)  # Evaluate at parameter value 0.5
    0.5

    Notes
    -----
    - Supports arbitrary knot vectors (non-uniform spacing)
    - Implements recursive Cox-de Boor algorithm
    - Automatically caches computed functions for performance
    - Handles repeated knots and various continuity conditions
    - Essential for NURBS and isogeometric analysis applications
    """
    registry = dict()

    @classmethod
    def list(cls):
        """
        List all cached NonUniformBSpline instances in a formatted table.

        Returns
        -------
        str
            Formatted table showing knot vector, degree, knot index, and
            derivative order of all cached B-spline functions.
        """
        text = cls.__name__ + "\n" + \
               tabulate([(k[0].name, k[1], k[2], k[3]) for (k, v) in cls.registry.items()],
                        headers=("T", "Degree", "Knot", "Derivative"), tablefmt='orgtbl')
        return text

    @property
    def key(self):
        """Unique identifier tuple (t.key, m, k, derivative) for caching."""
        return tuple(np.concatenate([self.t.key, self.m, self.k, self.derivative], axis=None))

    def __init__(self, t, m, k, name=None):
        """
        Initialize a Non-uniform B-spline basis function.

        Parameters
        ----------
        t : Knotvector
            Knot vector defining the parameter space.
        m : int
            Degree of the B-spline (must be > 0).
        k : int
            Index of the basis function within the knot vector.
        name : str, optional
            Optional name identifier.

        Raises
        ------
        AssertionError
            If m <= 0.

        Notes
        -----
        - First checks registry cache before computation
        - For m=1: creates constant basis function on [t_k, t_{k+1}]
        - For m>1: recursively builds from lower-degree B-splines using Cox-de Boor
        - Automatically stores computed functions in registry
        - Handles zero denominators in knot differences gracefully
        """
        self.t, self.m, self.k, self.derivative = t, m, k, 0

        # Retrive from registry.
        mycopy: NonUniformBSpline = NonUniformBSpline.retrieve(self.key)
        if mycopy is not None:
            self.build_from(mycopy)
            return

        # Otherwise, build from scratch, and store in registry.
        assert m > 0, "Degree must be larger than 0."

        if m == 1:
            tk = t[k]
            tk1 = t[k + 1]
            B = Function(Branch(numerator=Polynomial([1]), support=[tk, tk1]))
        else:
            # Extract knot values
            t_k = t[k]
            t_k1 = t[k + 1]
            t_km = t[k + m]
            t_km1 = t[k + m - 1]

            # Fraction 1
            den = t_km1 - t_k
            C1 = Function(Branch(numerator=Polynomial([0]), support=t.domain)) if den == 0 else \
                Function(Branch(numerator=Polynomial([1, -t_k]), denominator=Polynomial(den), support=t.domain))
            F1 = NonUniformBSpline(t, m-1, k)
            G1 = F1 * C1

            # Fraction 2
            den = t_km - t_k1
            C2 = Function(Branch(numerator=Polynomial(0), support=t.domain)) if den == 0 else \
                Function(Branch(numerator=Polynomial([-1, t_km]), denominator=Polynomial(den), support=t.domain))
            F2 = NonUniformBSpline(t, m-1, k+1)
            G2 = F2 * C2

            B = G1 + G2
            B.simplify()

        super().__init__(B.branches, name=name)
        NonUniformBSpline.store(key=self.key, item=self)

    def diff(self, n: int = 1, copy=True, name=None):
        """
        Compute the n-th derivative of the non-uniform B-spline basis function.

        Parameters
        ----------
        n : int, optional
            Order of derivative (default: 1).
        copy : bool, optional
            If True, returns a derivative copy; if False, differentiates in-place.
        name : str, optional
            Optional name for the derivative function.

        Returns
        -------
        NonUniformBSpline
            The n-th derivative as a new NonUniformBSpline object.

        Notes
        -----
        - Checks registry cache before computing derivatives
        - Automatically caches computed derivatives
        - Updates the derivative attribute accordingly
        - Uses the underlying Function differentiation capabilities
        """
        new_key = tuple(np.concatenate([self.t.key, self.m, self.k, self.derivative+n], axis=None))

        if copy:
            # Retrieval works with deepcopies, so it should be used if copy is False.
            mycopy = NonUniformBSpline.retrieve(new_key)

            if mycopy is not None:
                return mycopy
            else:
                dv = super().diff(n,copy, name)
                NonUniformBSpline.store(key=new_key, item=dv)
                return dv

        else:
            super().diff(n, copy, name)
            NonUniformBSpline.store(key=new_key, item=self)
            return self
