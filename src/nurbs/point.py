import numpy as np
from typing import Union, Iterable
import matplotlib.pylab as plt
from src.system import Copyable, NumericValues


class Point(Copyable):
    """
    A 3D point class with Euclidean coordinates and mathematical operations.

    This class represents a point in 2D or 3D space with support for vector
    operations, coordinate access, and visualization. It automatically handles
    2D points by setting the z-coordinate to zero.

    Parameters
    ----------
    coordinates : array_like
        Euclidean coordinates [x, y] or [x, y, z].
    name : str, optional
        Optional name identifier for the point.

    Attributes
    ----------
    coordinates : ndarray
        3D coordinate array [x, y, z]. For 2D points, z=0.
    name : str or None
        Optional name identifier.
    x : float
        X-coordinate (property).
    y : float
        Y-coordinate (property).
    z : float
        Z-coordinate (property).

    Examples
    --------
    >>> # Create 2D point
    >>> p1 = Point([1.0, 2.0], name="point_2d")
    >>> print(p1.x, p1.y, p1.z)
    1.0 2.0 0.0
    >>>
    >>> # Create 3D point
    >>> p2 = Point([1.0, 2.0, 3.0], name="point_3d")
    >>> p1 + p2  # Vector addition
    >>> 2 * p1   # Scalar multiplication

    Notes
    -----
    - 2D points are automatically extended to 3D with z=0
    - Supports all standard vector operations (addition, subtraction, scalar multiplication)
    - Supports matrix multiplication for affine transformations
    - Uses Copyable pattern for caching and memoization
    - Array priority ensures proper handling in mixed numpy operations
    """

    # Change array priority, so that A.x is respected and redirected in __rmul__ of Point.
    __array_priority__ = 10

    @classmethod
    def parse(cls, obj):
        """
        Convert various object types to Point instances.

        Parameters
        ----------
        obj : Point, NumericValues
            Object to convert to Point. Can be:
            - Existing Point instance (returned as-is)
            - Numeric value (converted to point [value, value, value])

        Returns
        -------
        Point
            Point representation of the input object.

        Raises
        ------
        RuntimeError
            If the object type cannot be converted to Point.
        """
        if isinstance(obj, cls):
            return obj
        elif isinstance(obj, NumericValues):
            return Point(coordinates=obj*np.ones(shape=(3,)))
        else:
            raise RuntimeError('Point.parse: unknown type ' + type(obj))

    def __init__(self, coordinates, name=None):
        """
        Initialize a Point with Euclidean coordinates.

        Parameters
        ----------
        coordinates : array_like
            Coordinate values. Can be 2D [x, y] or 3D [x, y, z].
        name : str, optional
            Optional name identifier for caching and display.

        Raises
        ------
        AssertionError
            If coordinates array has invalid size (not 2 or 3 elements).
        """

        # Convert to array in order to enable the use of ".size".
        coordinates = np.asarray(coordinates, dtype=float)

        # Make sure that at least one coordinate was provided.
        assert 1 <= coordinates.size <= 3, "Error in point definition-> Number of coordinates must be 2 or 3."

        # Initialize internal coordinates variable.
        self.coordinates = np.zeros(shape=(3,), dtype=float)
        self.coordinates[:coordinates.size] = coordinates[:]

        self.name = name

    def __str__(self):
        """Return string representation of coordinates."""
        return str(self.coordinates)

    def __repr__(self):
        """Return detailed representation including class name and coordinates."""
        text = f"[{self.__class__.__name__}] "
        text += self.name + " " if self.name is not None else ""
        text += "= "
        text += str(self)
        return text

    @property
    def x(self) -> float:
        """Get the x-coordinate."""
        return self.coordinates[0]

    @x.setter
    def x(self, x:float) -> None:
        """Set the x-coordinate."""
        self.coordinates[0] = x

    @property
    def y(self) -> float:
        """Get the y-coordinate."""
        return self.coordinates[1]

    @y.setter
    def y(self, y: float) -> None:
        """Set the y-coordinate."""
        self.coordinates[1] = y

    @property
    def z(self) -> float:
        """Get the z-coordinate."""
        return self.coordinates[2]

    @z.setter
    def z(self, z:float) -> None:
        """Set the z-coordinate."""
        self.coordinates[2] = z

    def __getitem__(self, item: int) -> float:
        """
        Get coordinate by index.

        Parameters
        ----------
        item : int
            Index of coordinate (0=x, 1=y, 2=z).

        Returns
        -------
        float
            Coordinate value at the specified index.
        """
        return self.coordinates[item]

    def __setitem__(self, key: int, value: float) -> None:
        """
        Set coordinate by index.

        Parameters
        ----------
        key : int
            Index of coordinate to set (0=x, 1=y, 2=z).
        value : float
            New coordinate value.
        """
        self.coordinates[key] = value

    def __add__(self, other):
        """
        Add another point or scalar to this point.

        Parameters
        ----------
        other : Point or NumericValues
            Point to add (vector addition) or scalar to add to all coordinates.

        Returns
        -------
        Point
            New point resulting from the addition.

        Examples
        --------
        >>> p1 = Point([1, 2, 3])
        >>> p2 = Point([4, 5, 6])
        >>> p1 + p2  # Vector addition: [5, 7, 9]
        >>> p1 + 2   # Scalar addition: [3, 4, 5]
        """
        other = Point.parse(other)
        mycopy = self.deepcopy()
        return Point(self.coordinates + other.coordinates)

    def __radd__(self, other):
        """Right addition for commutative operation support."""
        return self + other

    def __neg__(self):
        """Return the negative of the point (unary minus)."""
        return Point(-self.coordinates)

    def __sub__(self, other):
        """Subtract another point or scalar from this point."""
        other = Point.parse(other)
        return self + (-other)

    def __rsub__(self, other):
        """Right subtraction (other - self)."""
        return (-self) + other

    def __mul__(self, other):
        """
        Multiply point by scalar or point.

        Parameters
        ----------
        other : NumericValues or ndarray
            Scalar value or transformation matrix.

        Returns
        -------
        Point
            New point resulting from multiplication.

        Raises
        ------
        RuntimeError
            If operand type is not supported.

        Examples
        --------
        >>> p = Point([1, 2, 3])
        >>> p * 2          # Scalar multiplication: [2, 4, 6]
        >>> p * np.eye(3)  # Matrix multiplication (identity): [1, 2, 3]
        """
        mycopy = self.deepcopy()

        if isinstance(other, NumericValues):
            mycopy.coordinates *= other
        elif isinstance(other, np.ndarray):
            mycopy.coordinates = np.dot(mycopy.coordinates, other)
        else:
            raise RuntimeError(f"Point __rmul__: Operand not supported: {other} of type {type(other)}.")

    def __rmul__(self, other):
        """
        Multiply scalar or matrix with point from the left.

        Parameters
        ----------
        other : NumericValues or ndarray
            Scalar value or transformation matrix.

        Returns
        -------
        Point
            New point resulting from multiplication.

        Raises
        ------
        RuntimeError
            If operand type is not supported.

        Examples
        --------
        >>> p = Point([1, 2, 3])
        >>> 2 * p          # Scalar multiplication: [2, 4, 6]
        >>> np.eye(3) * p  # Matrix multiplication: [1, 2, 3]
        """

        mycopy = self.deepcopy()

        if isinstance(other, NumericValues):
            mycopy.coordinates *= other
        elif isinstance(other, np.ndarray):
            mycopy.coordinates = np.dot(other, mycopy.coordinates)
        else:
            raise RuntimeError(f"Point __rmul__: Operand not supported: {other} of type {type(other)}.")

        return mycopy

    def plot(self, *args, **kwargs):
        """
        Plot the point using matplotlib.

        Parameters
        ----------
        *args : tuple
            Additional positional arguments passed to plt.plot.
        **kwargs : dict
            Additional keyword arguments passed to plt.plot.

        Examples
        --------
        >>> p = Point([1, 2])
        >>> p.plot('ro', markersize=10)  # Red circle marker
        """
        plt.plot(self.coordinates[:], *args, **kwargs)