import numpy as np
from src.system import Copyable


class Container(Copyable):
    """
    A generic container class for storing indexed items with customizable keys.

    This class provides a flexible container that associates items with unique
    keys, allowing for efficient lookup, modification, and deletion operations.
    It serves as a base class for specialized containers like KnotVector and
    supports both sequential and arbitrary key indexing.

    Parameters
    ----------
    items : array_like
        The values to be stored in the container.
    keys : array_like, optional
        The indices/keys for each item. If None, sequential integers are used.
    first_index : int, optional
        The starting index for sequential keys (default: 0).
    name : str, optional
        Optional name identifier for the container.

    Attributes
    ----------
    items : ndarray
        Array of stored values.
    keys : ndarray
        Array of keys corresponding to each item.
    first_index : int
        The first key value in the container.
    name : str or None
        Optional name identifier.
    nr_items : int
        Number of items in the container (property).

    Examples
    --------
    >>> # Create container with sequential keys
    >>> c1 = Container([10, 20, 30], name="sequential")
    >>> print(c1[1])  # Access by key
    20
    >>>
    >>> # Create container with custom keys
    >>> c2 = Container([100, 200, 300], keys=['a', 'b', 'c'], name="custom")
    >>> print(c2['b'])  # Access by custom key
    200

    Notes
    -----
    - Keys must be unique and sortable
    - Supports the Copyable pattern for caching and memoization
    - Keys are maintained in sorted order for efficient lookup
    - Used as base class for KnotVector and other specialized containers
    """

    def __init__(self, items, keys=None, first_index=0, name=None):
        """
        Initialize a Container with items and optional custom keys.

        Parameters
        ----------
        items : array_like
            The values to be stored in the container.
        keys : array_like, optional
            Custom keys for each item. If None, sequential integers starting
            from first_index are used.
        first_index : int, optional
            Starting index for sequential keys (default: 0).
        name : str, optional
            Optional name identifier for caching and display.
        """
        self.items = np.asarray(items)
        self.keys = np.arange(self.items.size) + first_index if keys is None else keys

        self.first_index = self.keys[0]
        self.name = name

    @property
    def nr_items(self) -> int:
        """
        Get the number of items in the container.

        Returns
        -------
        int
            The number of items stored in the container.
        """
        return self.keys.size

    def __str__(self):
        """
        Return string representation as a dictionary.

        Returns
        -------
        str
            Dictionary-style string showing key-value pairs.
        """
        if self.keys.size == 0:
            return "empty"
        return str(dict(zip(self.keys, self.items)))

    def __repr__(self):
        """
        Return detailed representation including class name and contents.

        Returns
        -------
        str
            Detailed string representation with class name and contents.
        """
        text = f"[{self.__class__.__name__}] "
        text += self.name + " " if self.name is not None else ""
        text += f"= {str(self)}"

        return text

    def keys_to_idx(self, keys):
        """
        Convert keys to internal array indices.

        Uses binary search to efficiently locate the positions of keys
        in the internal arrays.

        Parameters
        ----------
        keys : array_like
            Keys to convert to indices.

        Returns
        -------
        ndarray
            Array of indices corresponding to the input keys.

        Examples
        --------
        >>> c = Container([10, 20, 30], keys=[5, 10, 15])
        >>> c.keys_to_idx([10, 15])
        array([1, 2])
        """
        keys = np.asarray(keys)
        idx = np.searchsorted(self.keys, keys)
        return idx

    def __getitem__(self, keys):
        """
        Get items by their keys.

        Parameters
        ----------
        keys : array_like
            Keys of the items to retrieve.

        Returns
        -------
        ndarray
            Array of items corresponding to the input keys.

        Examples
        --------
        >>> c = Container([10, 20, 30], keys=['a', 'b', 'c'])
        >>> c['b']
        20
        >>> c[['a', 'c']]
        array([10, 30])
        """
        idx = self.keys_to_idx(keys)
        return self.items[idx]

    def __setitem__(self, keys, values):
        """
        Set items by their keys.

        Parameters
        ----------
        keys : array_like
            Keys of the items to modify.
        values : array_like
            New values for the specified items.

        Examples
        --------
        >>> c = Container([10, 20, 30], keys=['a', 'b', 'c'])
        >>> c['b'] = 25
        >>> print(c)
        {'a': 10, 'b': 25, 'c': 30}
        """
        idx = self.keys_to_idx(keys)
        self.items[idx] = values

    def __delitem__(self, keys):
        """
        Delete items by their keys.

        Parameters
        ----------
        keys : array_like
            Keys of the items to delete.

        Returns
        -------
        Container
            Self with specified items removed (modified in-place).

        Examples
        --------
        >>> c = Container([10, 20, 30], keys=['a', 'b', 'c'])
        >>> del c['b']
        >>> print(c)
        {'a': 10, 'c': 30}
        """
        idx = self.keys_to_idx(keys)
        self.keys = np.delete(self.keys, idx)
        self.items = np.delete(self.items, idx)
        return self

    def append_items(self, keys, items, copy=False):
        """
        Append new items to the container.

        Parameters
        ----------
        keys : array_like
            Keys for the new items.
        items : array_like
            Values for the new items.
        copy : bool, optional
            If True, returns a modified copy; if False, modifies in-place.

        Returns
        -------
        Container
            Container with new items added. Returns copy if copy=True.

        Examples
        --------
        >>> c = Container([10, 20], keys=['a', 'b'])
        >>> c.append_items(['c', 'd'], [30, 40])
        >>> print(c)
        {'a': 10, 'b': 20, 'c': 30, 'd': 40}
        """
        mycopy = self.deepcopy() if copy else self
        keys, items = np.asarray(keys), np.asarray(items)

        self.keys = np.append(self.keys, keys)
        self.items = np.append(self.items, items)

        return mycopy