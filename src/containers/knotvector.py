import numpy as np
from src.containers.knotcontainer import Knotcontainer
from src.system import NumericValues


class Knotvector(Knotcontainer):
    """
    A specialized knot container for B-spline computations with sequential integer keys.

    This class represents a knot vector used in B-spline and NURBS formulations,
    providing methods for knot insertion, refinement, and span finding. It extends
    Knotcontainer with the requirement that keys must be sequential integers.

    Parameters
    ----------
    degree : int
        The polynomial degree of the B-spline basis functions.
    domain : tuple, optional
        The domain interval [a, b] for the knot vector (default: (0, 1)).
    first_index : int, optional
        The starting index for knot keys (default: 1).
    name : str, optional
        Optional name identifier for the knot vector.

    Attributes
    ----------
    degree : int
        Polynomial degree of the B-spline.
    domain : tuple
        Domain interval [a, b] of the knot vector.
    items : ndarray
        Array of knot values with clamped boundaries.
    keys : ndarray
        Sequential integer keys starting from first_index.
    first_index : int
        The first key value in the knot vector.
    name : str or None
        Optional name identifier.
    kmin : int
        Minimum valid function key (property).
    kmax : int
        Maximum valid function key (property).
    valid_function_keys : ndarray
        Array of keys that generate valid B-spline basis functions (property).
    key : tuple
        Unique identifier tuple (degree, knot_values) for caching (property).

    Examples
    --------
    >>> # Create a quadratic B-spline knot vector on [0, 1]
    >>> kv = Knotvector(degree=2, domain=(0, 1))
    >>> print(kv.items)  # Clamped knots: [0, 0, 0, 1, 1, 1]
    [0. 0. 0. 1. 1. 1.]
    >>> kv.find_span(0.3)  # Find span containing x=0.3
    2

    Notes
    -----
    - Knot vectors are initialized with clamped boundaries (repeated knots)
    - Supports knot insertion and various refinement strategies (h, p, bisect)
    - Provides span finding for B-spline basis function evaluation
    - Keys are always sequential integers for efficient indexing
    """

    def __init__(self, degree, domain=(0, 1), first_index=1, name=None):
        """
        Initialize a Knotvector with clamped boundary knots.

        Parameters
        ----------
        degree : int
            Polynomial degree of the B-spline basis functions.
        domain : tuple, optional
            Domain interval [a, b] (default: (0, 1)).
        first_index : int, optional
            Starting index for knot keys (default: 1).
        name : str, optional
            Optional name identifier for caching.

        Notes
        -----
        - Creates a clamped knot vector with (degree + 1) repeated knots at each boundary
        - Total number of knots: 2*(degree + 1)
        - Suitable for standard B-spline computations with C^(degree-1) continuity
        """
        self.degree = degree
        self.domain = domain
        self.name = name

        _left = np.repeat(domain[0], self.degree + 1)
        _right = np.repeat(domain[1], self.degree + 1)
        _knots = np.asfarray(np.concatenate([_left, _right], axis=None))
        super().__init__(items=_knots, first_index=first_index, name=name)

    def slice_to_keys(self, s):
        """
        Slices a Knotvector object to an array of keys.

        Follows standard Python slicing conventions where the stop index
        is excluded from the result.

        Parameters
        ----------
        s : slice
            Python slice object with start, stop, and step components.

        Returns
        -------
        ndarray
            Array of integer keys corresponding to the slice.

        Examples
        --------
        >>> kv = Knotvector(degree=1, first_index=0)
        >>> kv.slice_to_keys(slice(0, 4, 2))
        array([0, 2])
        """

        # Convert to indices
        start = self.keys[0] if s.start is None else s.start
        stop = self.keys[-1] + 1 if s.stop is None else s.stop
        step = 1 if s.step is None else s.step

        return np.arange(start=start, stop=stop, step=step, dtype=int)

    def __getitem__(self, keys):
        """
        Get knot values by key or slice.

        Supports both integer keys and Python slice notation for
        accessing multiple consecutive knots.

        Parameters
        ----------
        keys : int, slice, or array_like
            Key(s) of the knots to retrieve.

        Returns
        -------
        float or ndarray
            Knot value(s) corresponding to the input key(s).

        Examples
        --------
        >>> kv = Knotvector(degree=1)
        >>> kv[1]  # Single key
        0.0
        >>> kv[1:4]  # Slice
        array([0., 0., 1.])
        """
        if isinstance(keys, slice):
            keys = self.slice_to_keys(keys)

        return super(Knotvector, self).__getitem__(keys)

    def __setitem__(self, keys, values):
        """
        Set knot values by key or slice.

        Supports both individual key assignment and slice-based
        assignment for multiple consecutive knots.

        Parameters
        ----------
        keys : int, slice, or array_like
            Key(s) of the knots to modify.
        values : float or array_like
            New knot value(s). If scalar, broadcast to all keys.

        Examples
        --------
        >>> kv = Knotvector(degree=1)
        >>> kv[2] = 0.5  # Single assignment
        >>> kv[1:3] = 0.25  # Multiple assignment with broadcasting
        """
        if isinstance(keys, slice):
            keys = self.slice_to_keys(keys)

        if isinstance(values, NumericValues):
            values = np.ones_like(keys)*values

        super(Knotvector, self).__setitem__(keys, values)

    def insert_knots(self, values, copy=False):
        """
        Insert new knot values into the knot vector.

        Parameters
        ----------
        values : array_like
            Knot values to insert. Must lie within the domain.
        copy : bool, optional
            If True, returns a modified copy; if False, modifies in-place.

        Returns
        -------
        Knotvector
            Knot vector with inserted knots. Returns copy if copy=True.

        Raises
        ------
        AssertionError
            If any knot value lies outside the domain.

        Examples
        --------
        >>> kv = Knotvector(degree=1, domain=(0, 1))
        >>> kv.insert_knots([0.3, 0.7])
        >>> print(kv.items)
        [0.  0.  0.3 0.7 1.  1. ]
        """
        mycopy = self.deepcopy() if copy else self

        # Make sure that the new knots are inside the domain.
        for value in values:
            assert mycopy.domain[0] <= value <= mycopy.domain[1], f"Knot {value} not in domain."

        mycopy.items = np.sort(np.concatenate([mycopy.items, values]))
        mycopy.keys = np.arange(mycopy.items.size) + mycopy.first_index

        return mycopy

    def refine(self, nr_times=1, method='bisect', copy=False, name=None):
        """
        Refine the knot vector using specified method.

        Parameters
        ----------
        nr_times : int, optional
            Number of refinement iterations to apply (default: 1).
        method : str, optional
            Refinement method:
            - 'bisect': Divide each knot span in half
            - 'h': h-refinement (insert knots to preserve continuity)
            - 'p': p-refinement (increase degree and insert knots)
        copy : bool, optional
            If True, returns a refined copy; if False, refines in-place.
        name : str, optional
            New name for the refined knot vector.

        Returns
        -------
        Knotvector
            Refined knot vector. Returns copy if copy=True.

        Raises
        ------
        RuntimeError
            If an unknown refinement method is specified.

        Examples
        --------
        >>> kv = Knotvector(degree=1, domain=(0, 1))
        >>> kv_refined = kv.refine(method='bisect')
        >>> print(kv_refined.items)
        [0.  0.  0.  0.5 1.  1.  1. ]
        """
        # Determine which object to refine.
        mycopy = self.deepcopy() if copy else self

        if method == "bisect":
            for iteration in range(nr_times):
                knots = mycopy.items
                new_knots = []
                for i in range(knots.size-1):
                    dx = 0.5 * (knots[i + 1] - knots[i])
                    if dx:
                        new_knots.append(knots[i] + dx)
                mycopy.insert_knots(new_knots)

        elif method == 'h':
            for iteration in range(nr_times):
                knots = mycopy.items
                new_knots = []
                for i in range(knots.size-1):
                    dx = 0.5 * (knots[i + 1] - knots[i])
                    if dx:
                        # Insert each new knot p-times, to preserve C0 continuity.
                        new_knots.extend([knots[i] + dx]*self.degree)
                mycopy.insert_knots(new_knots)

        elif method == 'p':
            knots = mycopy.items
            new_knots = []
            unique_knots = list(np.unique(self.items))
            for iteration in range(nr_times):
                mycopy.degree += 1
                new_knots.extend(unique_knots)
            mycopy.insert_knots(new_knots)

        else:
            raise RuntimeError ('Unknown refining method ', method)

        if name is not None: mycopy.name = name
        return mycopy

    def find_span(self, x):
        """
        Find the knot span containing a given parameter value.

        Returns the key k such that x ∈ [t_k, t_{k+1}), where t_k
        are the knot values.

        Parameters
        ----------
        x : float
            Parameter value to locate.

        Returns
        -------
        int or None
            Key k of the left knot of the span, or None if x is outside domain.

        Examples
        --------
        >>> kv = Knotvector(degree=1, domain=(0, 1))
        >>> kv.find_span(0.3)
        2
        >>> kv.find_span(1.5)  # Outside domain
        None
        """
        for k in self.keys[:-1]:
            if self[k] <= x < self[k+1]: return k
        return None

    @property
    def kmin(self) -> int:
        """
        Get the minimum valid key for B-spline basis functions.

        Returns
        -------
        int
            The smallest key that generates a valid B-spline basis function.
        """
        return self.first_index

    @property
    def kmax(self) -> int:
        """
       Get the maximum valid key for B-spline basis functions.

       For a knot vector with n knots and degree p, the maximum valid
       key is n - p - 2.

       Returns
       -------
       int
           The largest key that generates a valid B-spline basis function.
       """
        return self.keys[-self.degree - 2]

    @property
    def valid_function_keys(self) -> np.ndarray:
        """
        Get all keys that generate valid B-spline basis functions.

        Returns
        -------
        ndarray
            Array of integer keys from kmin to kmax (inclusive).
        """
        return np.arange(self.kmin, self.kmax + 1)

    @property
    def key(self):
        """
        Get a unique identifier tuple for caching purposes.

        Returns
        -------
        tuple
            Tuple containing (degree, knot_values) for unique identification.
        """
        return tuple(np.concatenate([self.degree, self.items], axis=None))