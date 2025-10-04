from .point import Point


class ControlPoint(Point):
    """
        A control point with homogeneous coordinates for NURBS and rational curve computations.

        This class extends the Point class to include a weight component (w) used in
        homogeneous coordinate representations. Control points are essential for
        NURBS (Non-Uniform Rational B-Spline) formulations where weights control
        the influence of each point on the curve shape.

        Parameters
        ----------
        coordinates : array_like
            Euclidean coordinates [x, y] or [x, y, z].
        w : float
            Weight component for homogeneous coordinates.
        name : str, optional
            Optional name identifier for the control point.

        Attributes
        ----------
        coordinates : ndarray
            3D coordinate array [x, y, z]. For 2D points, z=0.
        w : float
            Weight component for homogeneous coordinates.
        name : str or None
            Optional name identifier.
        x : float
            X-coordinate (inherited from Point).
        y : float
            Y-coordinate (inherited from Point).
        z : float
            Z-coordinate (inherited from Point).

        Examples
        --------
        >>> # Create a weighted control point
        >>> cp = ControlPoint([1.0, 2.0], w=0.5, name="control_point")
        >>> print(cp.x, cp.y, cp.w)
        1.0 2.0 0.5
        >>>
        >>> # Convert to Euclidean point
        >>> p = cp.to_point()
        >>> print(p.x, p.y)
        2.0 4.0

        Notes
        -----
        - Mathematical operations (+, -, *) return Point objects, not ControlPoint objects
        - Weights are not preserved during arithmetic operations
        - Used primarily in NURBS and rational Bézier curve computations
        - The weight w represents the homogeneous coordinate component
        - to_point() performs the perspective division (x/w, y/w, z/w)
        """

    @classmethod
    def parse(cls, obj):
        """
        Convert Point objects to ControlPoint instances.

        Creates a ControlPoint from a Point object with default weight=1.
        This is useful for converting regular points to control points
        in NURBS computations.

        Parameters
        ----------
        obj : Point
            Point object to convert to ControlPoint.

        Returns
        -------
        ControlPoint
            New ControlPoint with same coordinates and weight=1.

        Raises
        ------
        RuntimeError
            If the object type cannot be converted to ControlPoint.

        Examples
        --------
        >>> p = Point([1, 2, 3])
        >>> cp = ControlPoint.parse(p)
        >>> print(cp.w)
        1.0
        """
        if isinstance(obj, Point):
            return ControlPoint(obj.coordinates, w=1, name=obj.name)
        else:
            raise RuntimeError('ControlPoint.parse: unknown type ' + type(obj))

    def __init__(self, coordinates, w, name=None):
        """
        Initialize a ControlPoint with coordinates and weight.

        Parameters
        ----------
        coordinates : array_like
            Euclidean coordinates [x, y] or [x, y, z].
        w : float
            Weight component for homogeneous coordinates.
        name : str, optional
            Optional name identifier.

        Warning
        -------
        Addition, subtraction, and multiplication operations are inherited
        from the Point class and do not preserve the weight component.
        The result of these operations will be a Point object, not a
        ControlPoint object.
        """
        super().__init__(coordinates, name)
        self.w = w

    def __str__(self):
        """
        Return string representation with weight.

        Returns
        -------
        str
            String in format "[x y z] / w" showing both coordinates and weight.
        """
        return super().__str__() + " / " + str(self.w)

    def to_point(self):
        """
        Convert to Euclidean Point by perspective division.

        Performs the homogeneous coordinate transformation by dividing
        all coordinates by the weight component. This converts from
        homogeneous coordinates to standard Euclidean coordinates.

        Returns
        -------
        Point
            New Point object with coordinates (x/w, y/w, z/w).

        Examples
        --------
        >>> cp = ControlPoint([2.0, 4.0, 6.0], w=2.0)
        >>> p = cp.to_point()
        >>> print(p.coordinates)
        [1. 2. 3.]

        Notes
        -----
        - This operation is essential for evaluating NURBS curves and surfaces
        - If w=0, the result is undefined (division by zero)
        - The original ControlPoint remains unchanged
        """
        return Point(coordinates=self.coordinates/self.w, name=self.name)
