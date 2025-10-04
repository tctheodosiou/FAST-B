import numpy as np
from collections import Counter
from src.containers import Container
from src.system import is_sorted


class Knotcontainer(Container):
    """
    A specialized container for knot values with integer keys in ascending order.

    This class extends the Container class to specifically handle knot values
    used in B-spline and NURBS computations. It enforces sorted integer keys
    and provides methods for analyzing knot multiplicities and distributions.

    Parameters
    ----------
    keys : array_like, optional
        Integer keys for the knots. Must be in ascending order.
    items : array_like, optional
        Knot values to store in the container.
    first_index : int, optional
        Starting index for sequential keys when keys=None (default: 0).
    name : str, optional
        Optional name identifier for the container.

    Attributes
    ----------
    items : ndarray
        Array of knot values.
    keys : ndarray
        Array of integer keys in ascending order.
    first_index : int
        The first key value in the container.
    name : str or None
        Optional name identifier.

    Examples
    --------
    >>> # Create knot container with custom keys
    >>> kc = Knotcontainer(keys=[0, 1, 2, 3], items=[0.0, 0.5, 0.5, 1.0])
    >>> kc.multiplicity(0.5)
    2

    Notes
    -----
    - Keys must be integers in strictly ascending order
    - Used as base class for KnotVector which requires sequential keys
    - Maintains knot values for B-spline basis function computations
    - Provides multiplicity analysis for knot insertion algorithms
    """

    def __init__(self, keys=None, items=None, first_index=0, name=None):
        """
        Initialize a Knotcontainer with knot values and integer keys.

        Parameters
        ----------
        keys : array_like, optional
            Integer keys for the knots. If None, sequential keys are generated.
        items : array_like, optional
            Knot values to store in the container.
        first_index : int, optional
            Starting index for sequential keys (default: 0).
        name : str, optional
            Optional name identifier for caching and display.

        Raises
        ------
        AssertionError
            If keys are not in ascending order.
        """
        super().__init__(keys=keys, items=items, first_index=first_index, name=name)
        assert is_sorted(self.keys), "Knotcontainer: Keys must be ascending."

    def find_key(self, val, option=None):
        """
        Find the key(s) associated with a given knot value.

        Parameters
        ----------
        val : float
            Knot value to search for.
        option : str, optional
            Search mode:
            - None: returns the first occurrence (default)
            - 'all': returns all occurrences

        Returns
        -------
        int or ndarray or None
            Key of the first occurrence, array of all keys, or None if not found.

        Examples
        --------
        >>> kc = Knotcontainer(items=[0.0, 0.5, 0.5, 1.0])
        >>> kc.find_key(0.5)
        1
        >>> kc.find_key(0.5, option='all')
        array([1, 2])
        """
        idx = np.flatnonzero(self.items == val) # Returns tuple
        if len(idx) == 0: return None
        return self.keys[idx[0]] if option is None else self.keys[idx]

    def multiplicity(self, val) -> int:
        """
        Get the total multiplicity of a knot value.

        The multiplicity is the total number of times the knot value
        appears in the container.

        Parameters
        ----------
        val : float
            Knot value to count.

        Returns
        -------
        int
            Total number of occurrences of the knot value.

        Examples
        --------
        >>> kc = Knotcontainer(items=[0.0, 0.5, 0.5, 1.0])
        >>> kc.multiplicity(0.5)
        2
        """
        return Counter(self.items)[val]

    def right_multiplicity(self, k) -> int:
        """
        Get the right multiplicity of a knot at key k.

        The right multiplicity is the number of knots equal to the knot at key k,
        but with keys greater than or equal to k.

        Parameters
        ----------
        k : int
            Key of the knot to analyze.

        Returns
        -------
        int
            Number of equal knots with keys >= k.

        Examples
        --------
        >>> kc = Knotcontainer(keys=[0, 1, 2, 3], items=[0.0, 0.5, 0.5, 1.0])
        >>> kc.right_multiplicity(1)  # Knots at keys 1 and 2 have value 0.5
        2
        >>> kc.right_multiplicity(2)  # Only knot at key 2 has value 0.5
        1
        """
        if k not in self.keys: return 0
        idx = np.flatnonzero(self.items == self[k])
        return np.flatnonzero(self.keys[idx] >= k).size

    def left_multiplicity(self, k) -> int:
        """
        Get the left multiplicity of a knot at key k.

        The left multiplicity is the number of knots equal to the knot at key k,
        but with keys less than or equal to k.

        Parameters
        ----------
        k : int
            Key of the knot to analyze.

        Returns
        -------
        int
            Number of equal knots with keys <= k.

        Examples
        --------
        >>> kc = Knotcontainer(keys=[0, 1, 2, 3], items=[0.0, 0.5, 0.5, 1.0])
        >>> kc.left_multiplicity(2)  # Knots at keys 1 and 2 have value 0.5
        2
        >>> kc.left_multiplicity(1)  # Only knot at key 1 has value 0.5
        1
        """
        if k not in self.keys: return 0
        idx = np.flatnonzero(self.items == self[k])
        return np.flatnonzero(self.keys[idx] <= k).size

    def append_knots(self, keys, knots, copy=False):
        """
        Append new knots to the container.

        Parameters
        ----------
        keys : array_like
            Integer keys for the new knots.
        knots : array_like
            Knot values to append.
        copy : bool, optional
            If True, returns a modified copy; if False, modifies in-place.

        Returns
        -------
        Knotcontainer
            Container with new knots added. Returns copy if copy=True.

        Raises
        ------
        AssertionError
            If keys and knots arrays have different sizes.

        Examples
        --------
        >>> kc = Knotcontainer(items=[0.0, 1.0])
        >>> kc.append_knots([2, 3], [1.5, 2.0])
        >>> print(kc.items)
        [0.0, 1.0, 1.5, 2.0]
        """
        mycopy = self.deepcopy() if copy else self
        keys, knots = np.asarray(keys), np.asarray(knots)

        assert(keys.size == knots.size), "Knotcontainer.append_knots: keys/knots size mismatch."

        if knots.size == 0:
            print(RuntimeWarning('Knotcontainer.append_knots: Empty knot-array provided. No change.'))
            return mycopy

        self.keys = np.append(self.keys, keys)
        self.items = np.append(self.items, knots)

        return mycopy