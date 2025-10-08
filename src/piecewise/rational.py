from src.system import Copyable, NumericValues
from scipy.integrate import quadrature
from src.piecewise.polynomial import Polynomial


class Rational(Copyable):
    """
    A class for rational functions, representing the ratio of two polynomials.

    This class implements mathematical operations for rational functions (P(x)/Q(x))
    where both numerator and denominator are Polynomial objects. It provides
    functionality for arithmetic operations, differentiation, integration,
    and various transformations while maintaining mathematical correctness.

    Parameters
    ----------
    numerator : Polynomial
        The numerator polynomial of the rational function.
    denominator : Polynomial, optional
        The denominator polynomial (default: constant polynomial 1).
    name : str, optional
        Optional name identifier for the rational function.

    Attributes
    ----------
    numerator : Polynomial
        Numerator polynomial of the rational function.
    denominator : Polynomial
        Denominator polynomial of the rational function.
    name : str or None
        Optional name identifier.

    Examples
    --------
    >>> # Create rational function (x² + 1) / (x - 1)
    >>> num = Polynomial([1, 0, 1])
    >>> den = Polynomial([1, -1])
    >>> r = Rational(num, den, name="example")
    >>> r(2)  # Evaluate at x=2
    5.0
    >>> r.diff()  # First derivative

    Notes
    -----
    - Uses Polynomial objects for numerator and denominator
    - Implements automatic type conversion for mixed operations
    - Provides numerical integration via scipy.quadrature
    - Supports the Copyable pattern for caching and memoization
    """

    @classmethod
    def parse(cls, obj):
        """
        Convert various object types to Rational instances.

        Provides flexible conversion from numeric types, Polynomial objects,
        and existing Rational instances to ensure consistent Rational handling
        in mathematical operations.

        Parameters
        ----------
        obj : Rational, NumericValues, Polynomial
            Object to convert to Rational. Can be:
            - Existing Rational instance (returned as-is)
            - Numeric value (converted to constant rational)
            - Polynomial (converted to rational with denominator 1)

        Returns
        -------
        Rational
            Rational representation of the input object.

        Raises
        ------
        RuntimeError
            If the object type cannot be converted to Rational.

        Examples
        --------
        >>> Rational.parse(5)  # Constant rational 5/1
        >>> Rational.parse(Polynomial([1, 2]))  # (x + 2)/1
        >>> Rational.parse(existing_rational)  # Returns same instance
        """

        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, NumericValues):
            return Rational(numerator=Polynomial(obj), denominator=Polynomial(1))
        elif isinstance(obj, Polynomial):
            return Rational(numerator=obj, denominator=Polynomial(1))
        else:
            raise RuntimeError('Rational.parse: unknown type ' + type(obj))

    def __init__(self, numerator, denominator=Polynomial(1), name=None):
        """
        Initialize Rational function with numerator and denominator polynomials.

        Parameters
        ----------
        numerator : Polynomial
            Numerator polynomial of the rational function.
        denominator : Polynomial, optional
            Denominator polynomial (default: constant polynomial 1).
        name : str, optional
            Optional name identifier for the rational function.

        Raises
        ------
        AssertionError
            If either numerator or denominator is not a Polynomial instance.
        """
        assert isinstance(numerator, Polynomial), "Numerator must be Polynomial."
        assert isinstance(denominator, Polynomial), "Denominator must be Polynomial."

        self.numerator = numerator
        self.denominator = denominator
        self.name = name

    def __str__(self):
        """Return string representation of the rational function."""
        return f"{self.numerator} / {self.denominator}"

    def __repr__(self):
        """Return detailed representation including class name and polynomials."""
        text = f"[{self.__class__.__name__}] "
        text += self.name + " " if self.name is not None else ""
        text += "= "
        text += "\n\tNumerator:" + repr(self.numerator)
        text += "\n\tDenominator:" + repr(self.denominator)
        return text

    def __call__(self, x):
        """
        Evaluate the rational function at given points.

        Parameters
        ----------
        x : array_like
            Input values where the rational function will be evaluated.

        Returns
        -------
        np.ndarray
            Evaluated values of the rational function.

        Examples
        --------
        >>> r = Rational(Polynomial([1, 0]), Polynomial([1, -1]))  # x/(x-1)
        >>> r(2)  # 2/(2-1) = 2.0
        2.0
        >>> r([0, 1, 2])  # Evaluate at multiple points
        array([0., inf, 2.])
        """
        return self.numerator(x) / self.denominator(x)

    # Define standard operators.
    def __add__(self, other):
        """Add rational function with another rational function or compatible type."""
        other = Rational.parse(other)
        numerator = self.numerator * other.denominator + self.denominator * other.numerator
        denominator = self.denominator * other.denominator
        return Rational(numerator, denominator)

    def __radd__(self, other):
        """Right addition for commutative operation support."""
        return self + other

    def __neg__(self):
        """Return negative of rational function (unary minus)."""
        return Rational(numerator=-self.numerator, denominator=self.denominator.deepcopy())

    def __sub__(self, other):
        """Subtract another rational function or compatible type."""
        return self + (-other)

    def __rsub__(self, other):
        """Right subtraction (other - self)."""
        return -self + other

    def __mul__(self, other):
        """Multiply rational function with another rational function or compatible type."""
        mycopy = self.deepcopy()
        other = Rational.parse(other)

        mycopy.numerator *= other.numerator
        mycopy.denominator *= other.denominator
        return mycopy

    def __rmul__(self, other):
        """Right multiplication for commutative operation support."""
        return self * other

    def __pow__(self, power):
        """Raise rational function to integer power."""
        mycopy = self.deepcopy()
        mycopy.numerator = mycopy.numerator ** power
        mycopy.denominator = mycopy.denominator ** power
        return mycopy

    def updown(self):
        """
        Return the reciprocal (inverse) of the rational function.

        Returns
        -------
        Rational
            New rational function representing 1/self.

        Examples
        --------
        >>> r = Rational(Polynomial([1, 0]), Polynomial([1, 1]))  # x/(x+1)
        >>> r_inv = r.updown()  # (x+1)/x
        """
        mycopy = self.deepcopy()
        nom, den = mycopy.numerator, mycopy.denominator
        mycopy.numerator, mycopy.denominator = den, nom
        return mycopy

    def __truediv__(self, other):
        """Divide rational function by another rational function or compatible type."""
        mycopy = self.deepcopy()
        other = Rational.parse(other)
        return mycopy * other.updown()

    def __rtruediv__(self, other):
        """Right division (other / self)."""
        mycopy = self.deepcopy().updown()
        return other * mycopy

    def shift(self, h, copy=True, name=None):
        """
        Return a shifted copy of the rational function.

        Applies the transformation f(x) → f(x + h) to both numerator and denominator.

        Parameters
        ----------
        h : float
            Shift amount to apply to the rational function.
        copy : bool, optional
            If True, returns a modified copy; if False, modifies in-place.
        name : str, optional
            New name for the shifted rational function.

        Returns
        -------
        Rational
            Shifted rational function.
        """
        mycopy = self.deepcopy() if copy else self
        mycopy.numerator.shift(h, copy=False)
        mycopy.denominator.shift(h, copy=False)
        if name is not None:
            mycopy.name = name
        return mycopy

    def scale(self, c, copy=True, name=None):
        """
        Return a scaled (dilated) copy of the rational function.

        Applies the transformation f(x) → f(c·x) to both numerator and denominator.

        Parameters
        ----------
        c : float
            Scaling factor to apply.
        copy : bool, optional
            If True, returns a modified copy; if False, modifies in-place.
        name : str, optional
            New name for the scaled rational function.

        Returns
        -------
        Rational
            Scaled rational function.
        """
        mycopy = self.deepcopy() if copy else self
        mycopy.numerator.scale(c, copy=False)
        mycopy.denominator.scale(c, copy=False)
        if name is not None:
            mycopy.name = name
        return mycopy

    def diff(self, nr_times=1, copy=True, name=None):
        """
        Return the n-th derivative of the rational function.

        Uses the quotient rule for differentiation:
        d/dx [N(x)/D(x)] = (N'(x)D(x) - N(x)D'(x)) / D(x)²

        Parameters
        ----------
        nr_times : int, optional
            Order of derivative (default: 1).
        copy : bool, optional
            If True, returns a modified copy; if False, modifies in-place.
        name : str, optional
            New name for the derivative.

        Returns
        -------
        Rational
            Derivative rational function.

        Examples
        --------
        >>> r = Rational(Polynomial([1, 0]), Polynomial([1, 1]))  # x/(x+1)
        >>> r_prime = r.diff()  # First derivative: 1/(x+1)²
        """
        mycopy = self.deepcopy() if copy else self
        nom, den = mycopy.numerator, mycopy.denominator
        for i in range(nr_times):
            nom, den = nom.deriv()*den - nom*den.deriv(), den ** 2
            # R = Rational(n.diff()*d - n*d.diff(), d**2) # Dont do this, because it reassigns R, and loses ref.
        mycopy.numerator, mycopy.denominator = nom, den
        if name is not None:
            mycopy.name = name
        return mycopy

    def integral(self, x1, x2, *args, **kwargs):
        """
        Compute definite integral of the rational function numerically.

        Uses scipy.integrate.quadrature for numerical integration.

        Parameters
        ----------
        x1 : float
            Lower integration limit.
        x2 : float
            Upper integration limit.
        *args : tuple
            Additional positional arguments passed to scipy.integrate.quadrature.
        **kwargs : dict
            Additional keyword arguments passed to scipy.integrate.quadrature.

        Returns
        -------
        tuple
            (integral_value, error_estimate) as returned by scipy.integrate.quadrature.

        Examples
        --------
        >>> r = Rational(Polynomial([1]), Polynomial([1, 1]))  # 1/(x+1)
        >>> integral, error = r.integral(0, 1)  # ∫₀¹ 1/(x+1) dx = ln(2)
        """
        return quadrature(lambda x: self(x), x1, x2, *args, **kwargs)

    def simplify(self):
        """
        Simplify the rational function by removing common factors.

        Currently removes trailing zeros from both numerator and denominator
        polynomials and normalizes by the leading coefficient of the denominator.

        Returns
        -------
        Rational
            Simplified rational function (modified in-place).

        Notes
        -----
        - This is a basic simplification that removes common factors of x
        - More sophisticated polynomial GCD simplification could be added
        - Normalization ensures denominator leading coefficient is 1
        """
        if self.numerator.coeffs[-1] == 0. and self.denominator.coeffs[-1] == 0.:
            self.numerator = Polynomial(self.numerator.coeffs[0:-1])
            self.denominator = Polynomial(self.denominator.coeffs[0:-1])

        self.numerator /= self.denominator[0]
        self.denominator /= self.denominator[0]
        return self