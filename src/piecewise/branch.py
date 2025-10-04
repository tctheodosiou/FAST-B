import numpy as np
import operator
from src.system import NumericValues
from src.piecewise import Polynomial, Rational


class Branch(Rational):
    """
        A piecewise rational function defined on a specific interval (branch).

        This class extends the Rational class to represent rational functions
        that are defined only on a specific support interval with customizable
        boundary inclusion. It implements mathematical operations that respect
        the domain restrictions and automatically handles support intersections.

        Parameters
        ----------
        support : array_like
            The domain interval [a, b] where the branch is defined.
        numerator : Polynomial
            Numerator polynomial of the rational function.
        denominator : Polynomial, optional
            Denominator polynomial (default: constant polynomial 1).
        includes_left_boundary : bool, optional
            Whether the left boundary is included in the domain (default: True).
        includes_right_boundary : bool, optional
            Whether the right boundary is included in the domain (default: False).
        name : str, optional
            Optional name identifier for the branch.

        Attributes
        ----------
        support : ndarray
            The domain interval [a, b] as a numpy array.
        includes_left_boundary : bool
            Flag indicating if left boundary is included.
        includes_right_boundary : bool
            Flag indicating if right boundary is included.
        ops : dict
            Dictionary containing comparison operators for boundary checks.
        domain : str
            String representation of the domain with boundary notation.
        midpoint : float
            Midpoint of the support interval.
        formula : Rational
            The underlying rational function without domain restrictions.

        Examples
        --------
        >>> # Create a branch: (x² + 1)/(x - 1) defined on [0, 2)
        >>> num = Polynomial([1, 0, 1])
        >>> den = Polynomial([1, -1])
        >>> branch = Branch(support=[0, 2], numerator=num, denominator=den,
        ...                includes_right_boundary=False)
        >>> branch(1.5)  # Evaluate within domain
        3.25
        >>> branch(2.0)  # Outside domain (right boundary excluded)
        0.0

        Notes
        -----
        - Boundary notation: [a,b] includes both, (a,b) excludes both,
          [a,b) includes left only, (a,b] includes right only
        - Mathematical operations automatically compute intersection of supports
        - Evaluation returns 0 outside the defined domain
        - Inherits all Rational functionality with domain awareness
        """

    @classmethod
    def parse(cls, obj):
        """
        Convert various object types to Branch instances.

        Provides flexible conversion from numeric types, Polynomial objects,
        and existing Branch instances. For simple types, creates a branch
        with infinite support.

        Parameters
        ----------
        obj : Branch, NumericValues, Polynomial
            Object to convert to Branch. Can be:
            - Existing Branch instance (returned as-is)
            - Numeric value (converted to constant branch on (-∞, ∞))
            - Polynomial (converted to branch on (-∞, ∞))

        Returns
        -------
        Branch
            Branch representation of the input object.

        Raises
        ------
        RuntimeError
            If the object type cannot be converted to Branch.

        Examples
        --------
        >>> Branch.parse(5)  # Constant branch 5/1 on (-∞, ∞)
        >>> Branch.parse(Polynomial([1, 2]))  # (x+2)/1 on (-∞, ∞)
        """

        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, NumericValues):
            return Branch(numerator=Polynomial(obj), denominator=Polynomial(1), support=(-np.inf, np.inf))
        elif isinstance(obj, Polynomial):
            return Branch(numerator=obj, denominator=Polynomial(1), support=(-np.inf, np.inf))
        else:
            raise RuntimeError('Branch.parse: unknown type ' + type(obj))

    @classmethod
    def common_support(cls, b1, b2):
        """
        Compute the intersection support of two Branch objects.

        Finds the common domain where both branches are defined and
        determines boundary inclusion based on both branches' settings.

        Parameters
        ----------
        b1 : Branch
            First branch object.
        b2 : Branch
            Second branch object.

        Returns
        -------
        tuple
            A tuple containing:
            - support: (float, float) the intersection interval
            - inc_left_boundary: bool whether left boundary is included
            - inc_right_boundary: bool whether right boundary is included

        Notes
        -----
        - Returns (0, 0) if there is no intersection
        - Boundary is included only if both branches include it
        """
        x1 = np.max([b1.support[0], b2.support[0]])
        x2 = np.min([b1.support[1], b2.support[1]])
        inc_left_boundary = b1.includes(x1) and b2.includes(x1)
        inc_right_boundary = b1.includes(x2) and b2.includes(x2)

        if x1 > x2:
            x1, x2 = 0, 0 # Boundary mismatch.
        return (x1, x2), inc_left_boundary, inc_right_boundary

    def __init__(self, support, numerator, denominator=Polynomial(1),
                 includes_left_boundary=True, includes_right_boundary=False,
                 name=None):
        """
        Initialize Branch with domain restrictions.

        Parameters
        ----------
        support : array_like
            The domain interval [a, b] where the branch is defined.
        numerator : Polynomial
            Numerator polynomial of the rational function.
        denominator : Polynomial, optional
            Denominator polynomial (default: constant polynomial 1).
        includes_left_boundary : bool, optional
            Whether left boundary is included (default: True).
        includes_right_boundary : bool, optional
            Whether right boundary is included (default: False).
        name : str, optional
            Optional name identifier.

        Raises
        ------
        AssertionError
            If support does not have exactly 2 elements.
        """

        super().__init__(numerator, denominator, name)

        self.support = np.asarray(support, dtype=float)
        assert self.support.size == 2, f"BRANCH INIT: Invalid support provided: {self.support}"

        self.includes_left_boundary = includes_left_boundary
        self.includes_right_boundary = includes_right_boundary

    @property
    def ops(self):
        """
        Get boundary comparison operators.

        Returns operators for checking if a point is within the domain,
        considering boundary inclusion settings.

        Returns
        -------
        dict
            Dictionary with keys:
            - "left": operator for left boundary check (≤ or <)
            - "right": operator for right boundary check (≤ or <)
        """
        return dict({
            "left": operator.le if self.includes_left_boundary else operator.lt,
            "right": operator.le if self.includes_right_boundary else operator.lt
            }
        )

    # Define properties
    @property
    def domain(self):
        """
        Get string representation of the domain with boundary notation.

        Returns
        -------
        str
            Domain in interval notation, e.g., "[0,1)", "(2,5]", etc.
        """
        left = "[" if self.includes_left_boundary else "("
        right = "]" if self.includes_right_boundary else ")"
        return f"{left}{self.support[0]},{self.support[1]}{right}"

    @property
    def midpoint(self):
        """
        Get the midpoint of the support interval.

        Returns
        -------
        float
            The midpoint (a + b)/2 of the support interval.
        """
        return 0.5 * (self.support[0] + self.support[1])

    @property
    def formula(self):
        """
        Get the underlying rational function without domain restrictions.

        Returns
        -------
        Rational
            The rational function that defines this branch, independent
            of the domain restrictions.
        """
        return Rational(numerator=self.numerator, denominator=self.denominator)

    def __str__(self):
        """Return string representation with domain information."""
        return f"{super().__str__()} on {self.domain}"

    def __repr__(self):
        """Return detailed representation including domain."""
        return f"{super().__repr__()}".replace(f"[{self.__class__.__name__}] =",
                                               f"[{self.__class__.__name__}] on {self.domain} =")

    def __call__(self, x):
        """
       Evaluate the branch function at given points.

       Returns the rational function value for points within the domain
       and 0 for points outside the domain.

       Parameters
       ----------
       x : array_like
           Input values where the branch will be evaluated.

       Returns
       -------
       ndarray
           Evaluated values. Zero outside the domain, rational function
           value inside the domain.
       """
        left_bc = np.asarray(self.ops["left"](self.support[0], x), dtype=float)
        right_bc = np.asarray(self.ops["right"](x, self.support[1]), dtype=float)
        return left_bc * super().__call__(x) * right_bc

    def includes(self, x):
        """
        Check if a point is within the branch's domain.

        Parameters
        ----------
        x : NumericValues
            Point to check for domain inclusion.

        Returns
        -------
        bool
            True if the point is within the domain (considering boundaries),
            False otherwise.
        """
        return self.ops["left"](self.support[0], x) and \
               self.ops["right"](x, self.support[1])

    # Define standard operators.
    def __add__(self, other):
        """Add two branches, computing intersection of supports."""
        other = Branch.parse(other)
        support, inc_left, inc_right = Branch.common_support(self, other)
        R = self.formula + other.formula
        return Branch(numerator=R.numerator, denominator=R.denominator, support=support,
                      includes_left_boundary=inc_left, includes_right_boundary=inc_right)

    def __radd__(self, other):
        """Right addition for commutative operation support."""
        return self + other

    def __neg__(self):
        """Return negative of the branch."""
        mycopy = self.deepcopy()
        mycopy.numerator = -mycopy.numerator
        return mycopy

    def __sub__(self,other):
        """Subtract another branch from this one."""
        other = Branch.parse(other)
        support, inc_left, inc_right = Branch.common_support(self, other)
        R = self.formula + (-other.formula)
        return Branch(numerator=R.numerator, denominator=R.denominator, support=support,
                      includes_left_boundary=inc_left, includes_right_boundary=inc_right)

    def __rsub__(self, other):
        """Right subtraction (other - self)."""
        return -self + other

    def __mul__(self, other):
        """Multiply this branch by another branch or compatible type."""
        other = Branch.parse(other)
        support, inc_left, inc_right = Branch.common_support(self, other)
        R = self.formula * other.formula
        return Branch(numerator=R.numerator, denominator=R.denominator, support=support,
                      includes_left_boundary=inc_left, includes_right_boundary=inc_right)

    def __rmul__(self, other):
        """Right multiplication for commutative operation support."""
        return self * other

    def __truediv__(self, other):
        """Divide this branch by another branch or compatible type."""
        other = Branch.parse(other)
        support, inc_left, inc_right = Branch.common_support(self, other)
        R = self.formula / other.formula
        return Branch(numerator=R.numerator, denominator=R.denominator, support=support,
                      includes_left_boundary=inc_left, includes_right_boundary=inc_right)

    def __rtruediv__(self, other):
        """Right division (other / self)."""
        other = Branch.parse(other)
        support, inc_left, inc_right = Branch.common_support(self, other)
        R = other.formula / self.formula
        return Branch(numerator=R.numerator, denominator=R.denominator, support=support,
                      includes_left_boundary=inc_left, includes_right_boundary=inc_right)

    def shift(self, h, copy=True, name=None):
        """
        Shift the branch horizontally.

        Applies the transformation f(x) → f(x - h) and shifts the support
        accordingly. Positive h shifts to the right.

        Parameters
        ----------
        h : float
            Shift amount. Positive values shift right.
        copy : bool, optional
            If True, returns a modified copy; if False, modifies in-place.
        name : str, optional
            New name for the shifted branch.

        Returns
        -------
        Branch
            Shifted branch.
        """
        mycopy = super().shift(h, copy, name)
        mycopy.support = mycopy.support - h
        return mycopy

    def scale(self, c, copy=True, name=None):
        """
        Scale the branch horizontally.

        Applies the transformation f(x) → f(c·x) and scales the support
        accordingly.

        Parameters
        ----------
        c : float
            Scaling factor.
        copy : bool, optional
            If True, returns a modified copy; if False, modifies in-place.
        name : str, optional
            New name for the scaled branch.

        Returns
        -------
        Branch
            Scaled branch.
        """
        mycopy = super(Branch, self).scale(c, copy, name)
        mycopy.support = mycopy.support / c
        return mycopy

    def integral(self, x1, x2, *args, **kwargs):
        """
        Compute definite integral over the intersection with [x1, x2].

        Integrates the branch function over the portion of [x1, x2] that
        lies within the branch's support.

        Parameters
        ----------
        x1 : float
            Lower integration limit.
        x2 : float
            Upper integration limit.
        *args : tuple
            Additional arguments passed to scipy.integrate.quadrature.
        **kwargs : dict
            Additional keyword arguments passed to scipy.integrate.quadrature.

        Returns
        -------
        tuple
            (integral_value, error_estimate) from scipy.integrate.quadrature.
            Returns (0, 0) if there is no intersection.
        """
        if x1 >= self.support[1] or x2 <= self.support[0]:
            # If the support is not valid, set the two ends equal, so that the parent function returns 0, upon call.
            x1 = x2
        else:
            x1 = np.max([self.support[0], x1])
            x2 = np.min([self.support[1], x2])
        return super().integral(x1, x2, *args, **kwargs)

    @property
    def extreme(self):
        """
        Find the extreme value of the branch within its support.

        Computes the maximum absolute value within the domain by checking:
        - Roots of the derivative (critical points)
        - Boundary points (considering inclusion)

        Returns
        -------
        float
            The extreme value with largest magnitude (positive or negative).

        Notes
        -----
        - Used primarily for normalization purposes
        - Considers both boundaries regardless of inclusion settings
        - Returns the value with largest absolute magnitude
        """
        # Work on a copy, to include boundaries.
        mycopy = self.deepcopy()
        mycopy.includes_left_boundary = True
        mycopy.includes_right_boundary = True

        # Find roots of the derivative, i.e. candidate locations for
        # extreme values.
        dB = mycopy.diff()
        derivative_nominator_polynomial = dB.numerator
        R = np.roots(derivative_nominator_polynomial)

        # Keep only x-values within the support.
        R = R[R > self.support[0]]
        R = R[R < self.support[1]]

        # Also add support values
        R = np.append(R, self.support)

        candidate_extremes = mycopy(R)
        positive_extreme = np.max(candidate_extremes).item()
        negative_extreme = np.min(candidate_extremes).item()

        y0 = positive_extreme \
            if positive_extreme > np.abs(negative_extreme) \
            else negative_extreme
        return y0
