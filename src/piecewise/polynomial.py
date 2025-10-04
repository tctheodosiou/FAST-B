import numpy as np
from src.system import Copyable, NumericValues


class Polynomial(np.poly1d, Copyable):
    """
    An enhanced polynomial class extending numpy.poly1d with additional functionality.

    This class combines the numerical capabilities of numpy.poly1d with the
    copying and caching features of the Copyable mixin. It provides enhanced
    polynomial operations, type conversion, and mathematical transformations
    while maintaining compatibility with standard numpy polynomial operations.

    Parameters
    ----------
    coefficients : array_like
        Polynomial coefficients in decreasing order of power (same as numpy.poly1d).
        For example, [1, 2, 3] represents x² + 2x + 3.
    name : str, optional
        Optional name identifier for the polynomial, useful for debugging and display.

    Attributes
    ----------
    coeffs : ndarray
        Polynomial coefficients (inherited from numpy.poly1d).
    name : str or None
        Optional name identifier for the polynomial.

    Examples
    --------
    >>> p = Polynomial([1, 2, 3], name="quadratic")
    >>> print(p)
    [1. 2. 3.]
    >>> p(2)  # Evaluate at x=2
    11.0
    >>> q = p.shift(1)  # Shift polynomial by 1
    >>> p + q  # Polynomial addition
    >>> p.diff()  # Derivative

    Notes
    -----
    - Inherits from both numpy.poly1d and Copyable for combined functionality
    - Automatically converts coefficients to float type for numerical stability
    - Provides safe operator overloading with type conversion
    - Polynomial division is intentionally prohibited; instead the "Rational" object should be empolyed
    """

    @classmethod
    def parse(cls, obj):
        """
        Convert various object types to Polynomial instances.

        This method provides flexible conversion from numeric types and
        existing Polynomial objects to ensure consistent Polynomial handling
        in mathematical operations.

        Parameters
        ----------
        obj : Polynomial, NumericValues
            Object to convert to Polynomial. Can be:
            - Existing Polynomial instance (returned as-is)
            - Numeric value (converted to constant polynomial)

        Returns
        -------
        Polynomial
            Polynomial representation of the input object.

        Raises
        ------
        RuntimeError
            If the object type cannot be converted to Polynomial.

        Examples
        --------
        >>> Polynomial.parse(5)  # Constant polynomial
        >>> Polynomial.parse(np.array([1, 2, 3]))  # Coefficient array
        >>> Polynomial.parse(existing_polynomial)  # Returns same instance
        """
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, NumericValues):
            return Polynomial(coefficients=np.asarray(obj))
        else:
            raise RuntimeError('Polynomial.parse: unknown type ' + type(obj))

    @classmethod
    def parse_poly1d(cls, obj):
        """
        Convert objects to numpy.poly1d instances for internal operations.

        This method is used internally to ensure compatibility with numpy's
        polynomial operations while maintaining the enhanced Polynomial interface.

        Parameters
        ----------
        obj : Polynomial, numpy.poly1d, NumericValues
            Object to convert to numpy.poly1d.

        Returns
        -------
        numpy.poly1d
            poly1d representation of the input object.

        Raises
        ------
        RuntimeError
            If the object type cannot be converted to poly1d.
        """
        if isinstance(obj, np.poly1d):
            return np.poly1d(obj.coeffs)
        elif isinstance(obj, NumericValues):
            return np.poly1d(obj)
        else:
            raise RuntimeError('Polynomial.parse_poly: unknown type ' + type(obj))

    def __init__(self, coefficients, name=None):
        """
        Initialize Polynomial instance with coefficients and optional name.

        Parameters
        ----------
        coefficients : array_like
            Polynomial coefficients in decreasing order of power.
            Converted to float array for numerical stability.
        name : str, optional
            Optional identifier for the polynomial.
        """
        super().__init__(np.asarray(coefficients, dtype=float))
        self.name = name

    def __str__(self):
        """Return string representation of polynomial coefficients."""
        return str(self.coeffs)

    def __repr__(self):
        """Return detailed representation including class name and optional name."""
        text = f"[{self.__class__.__name__}] "
        text += self.name + " " if self.name is not None else ""
        text += "= " + str(self)
        return text

    # Define standard operators.
    def __add__(self, other):
        """Add polynomial with another polynomial or numeric value."""
        other = Polynomial.parse_poly1d(other)
        return Polynomial(super().__add__(other).coeffs)

    def __radd__(self, other):
        """Right addition for commutative operation support."""
        other = Polynomial.parse_poly1d(other)
        return self + other

    def __neg__(self):
        """Return negative of polynomial (unary minus)."""
        return Polynomial(-super().coeffs)

    def __sub__(self, other):
        """Subtract another polynomial or numeric value."""
        other = Polynomial.parse_poly1d(other)
        return Polynomial(super().__sub__(other).coeffs)

    def __rsub__(self, other):
        """Right subtraction (other - self)."""
        other = Polynomial.parse_poly1d(other)
        return self + (-other)

    def __mul__(self, other):
        """Multiply polynomial with another polynomial or numeric value."""
        other = Polynomial.parse_poly1d(other)
        return Polynomial(super().__mul__(other).coeffs)

    def __rmul__(self, other):
        """Right multiplication for commutative operation support."""
        other = Polynomial.parse_poly1d(other)
        return self * other

    def __pow__(self, power):
        """Raise polynomial to integer power."""
        return Polynomial(super().__pow__(power).coeffs)

    def __truediv__(self, other):
        """
        Divide polynomial by numeric value (polynomial division prohibited).
        Raises
        ------
        RuntimeError
            If attempting polynomial division (use Rational class instead).
        """
        if isinstance(other, Polynomial):
            raise RuntimeError('Polynomial division is intentionally prohibited. Use a Rational instead.')

        return Polynomial(super().__truediv__(other).coeffs)

    def __rtruediv__(self, other):
        """
        Right division (other / self).

        This implements the operation when a numeric value is divided by a polynomial,
        which typically results in a rational function. Since polynomial division
        is prohibited, this operation is also restricted.

        Raises
        ------
        RuntimeError
            Always raised since polynomial division is intentionally prohibited.
        """
        raise RuntimeError('Division by polynomial is intentionally prohibited. Use a Rational instead.')

    # Define other operations.

    def shift(self, h, copy=True, name=None):
        """
        Return a shifted copy of the polynomial using Taylor shifting.

        Applies Taylor series expansion to shift the polynomial by `h` units.
        The operation computes p(x + h) using polynomial derivatives.

        Parameters
        ----------
        h : float
            Shift amount to apply to the polynomial.
        copy : bool, optional
            If True, returns a modified copy; if False, modifies in-place.
        name : str, optional
            New name for the shifted polynomial.

        Returns
        -------
        Polynomial
            Shifted polynomial instance.

        Notes
        -----
        - Uses Taylor expansion: p(x+h) = Σ [p⁽ⁿ⁾(x) * hⁿ / n!]
        - Known issue: For large shifts or high-order polynomials, numerical stability may vary
        """
        mycopy = self.deepcopy() if copy else self
        p0 = Polynomial.parse_poly1d(mycopy)
        p = np.poly1d(p0) # Initialize
        for n in range(1, mycopy.order + 1):
            p = np.polyadd(p, np.polyder(p0, n) * (h) ** n / np.math.factorial(n))
        mycopy._coeffs = p.coefficients

        if name is not None:
            mycopy.name = name

        return mycopy

    def scale(self, h, copy=True, name=None):
        """
        Return a scaled (dilated) copy of the polynomial.

        Applies scaling transformation p(x) → p(h·x) by adjusting coefficients.

        Parameters
        ----------
        h : float
            Scaling factor to apply.
        copy : bool, optional
            If True, returns a modified copy; if False, modifies in-place.
        name : str, optional
            New name for the scaled polynomial.

        Returns
        -------
        Polynomial
            Scaled polynomial instance.

        Examples
        --------
        >>> p = Polynomial([1, 0, 0])  # x²
        >>> p.scale(2)  # 4x² (since (2x)² = 4x²)
        """
        mycopy = self.deepcopy() if copy else self
        C = [h ** n for n in range(self.order, -1, -1)]
        mycopy._coeffs = mycopy._coeffs * C

        if name is not None:
            mycopy.name = name

        return mycopy

    def diff(self, n=1, copy=True, name=None):
        """
        Return the derivative of the polynomial.

        Computes the n-th derivative of the polynomial using numpy's
        polynomial differentiation.

        Parameters
        ----------
        n : int, optional
            Order of derivative (default: 1).
        copy : bool, optional
            If True, returns a modified copy; if False, modifies in-place.
        name : str, optional
            New name for the derivative polynomial.

        Returns
        -------
        Polynomial
            Derivative polynomial instance.
        """
        mycopy = self.deepcopy() if copy else self
        mycopy._coeffs = Polynomial.parse_poly1d(mycopy).deriv(n).coeffs
        if name is not None:
            mycopy.name = name
        return mycopy
